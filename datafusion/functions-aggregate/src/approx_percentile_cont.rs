// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::fmt::Debug;
use std::mem::{size_of, size_of_val};
use std::sync::Arc;

use arrow::array::{
    Array, AsArray, BooleanArray, Float16Array, Float64Builder, ListBuilder,
    UInt64Builder,
};
use arrow::compute::{filter, is_not_null};
use arrow::datatypes::{FieldRef, Float16Type, Float32Type, Float64Type, UInt64Type};
use arrow::{
    array::{ArrayRef, Float32Array, Float64Array},
    datatypes::{DataType, Field},
};
use datafusion_common::types::{NativeType, logical_float64};
use datafusion_common::{
    DataFusionError, Result, ScalarValue, downcast_value, internal_err, not_impl_err,
    plan_err,
};
use datafusion_expr::expr::{AggregateFunction, Sort};
use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion_expr::utils::format_state_name;
use datafusion_expr::{
    Accumulator, AggregateUDFImpl, Coercion, Documentation, EmitTo, Expr,
    GroupsAccumulator, Signature, TypeSignature, TypeSignatureClass, Volatility,
};
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::accumulate::accumulate;
use datafusion_functions_aggregate_common::aggregate::groups_accumulator::nulls::filtered_null_mask;
use datafusion_functions_aggregate_common::tdigest::{
    Centroid, DEFAULT_MAX_SIZE, TDigest,
};
use datafusion_macros::user_doc;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;

use crate::utils::{get_scalar_value, validate_percentile_expr};

create_func!(ApproxPercentileCont, approx_percentile_cont_udaf);

/// Computes the approximate percentile continuous of a set of numbers
pub fn approx_percentile_cont(
    order_by: Sort,
    percentile: Expr,
    centroids: Option<Expr>,
) -> Expr {
    let expr = order_by.expr.clone();

    let args = if let Some(centroids) = centroids {
        vec![expr, percentile, centroids]
    } else {
        vec![expr, percentile]
    };

    Expr::AggregateFunction(AggregateFunction::new_udf(
        approx_percentile_cont_udaf(),
        args,
        false,
        None,
        vec![order_by],
        None,
    ))
}

#[user_doc(
    doc_section(label = "Approximate Functions"),
    description = "Returns the approximate percentile of input values using the t-digest algorithm.",
    syntax_example = "approx_percentile_cont(percentile [, centroids]) WITHIN GROUP (ORDER BY expression)",
    sql_example = r#"```sql
> SELECT approx_percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) FROM table_name;
+------------------------------------------------------------------+
| approx_percentile_cont(0.75) WITHIN GROUP (ORDER BY column_name) |
+------------------------------------------------------------------+
| 65.0                                                             |
+------------------------------------------------------------------+
> SELECT approx_percentile_cont(0.75, 100) WITHIN GROUP (ORDER BY column_name) FROM table_name;
+-----------------------------------------------------------------------+
| approx_percentile_cont(0.75, 100) WITHIN GROUP (ORDER BY column_name) |
+-----------------------------------------------------------------------+
| 65.0                                                                  |
+-----------------------------------------------------------------------+
```
An alternate syntax is also supported:
```sql
> SELECT approx_percentile_cont(column_name, 0.75) FROM table_name;
+-----------------------------------------------+
| approx_percentile_cont(column_name, 0.75)     |
+-----------------------------------------------+
| 65.0                                          |
+-----------------------------------------------+

> SELECT approx_percentile_cont(column_name, 0.75, 100) FROM table_name;
+----------------------------------------------------------+
| approx_percentile_cont(column_name, 0.75, 100)           |
+----------------------------------------------------------+
| 65.0                                                     |
+----------------------------------------------------------+
```
"#,
    standard_argument(name = "expression",),
    argument(
        name = "percentile",
        description = "Percentile to compute. Must be a float value between 0 and 1 (inclusive)."
    ),
    argument(
        name = "centroids",
        description = "Number of centroids to use in the t-digest algorithm. _Default is 100_. A higher number results in more accurate approximation but requires more memory."
    )
)]
#[derive(Debug, PartialEq, Eq, Hash)]
pub struct ApproxPercentileCont {
    signature: Signature,
}

impl Default for ApproxPercentileCont {
    fn default() -> Self {
        Self::new()
    }
}

impl ApproxPercentileCont {
    /// Create a new [`ApproxPercentileCont`] aggregate function.
    pub fn new() -> Self {
        // Accept any numeric value paired with a float64 percentile
        let signature = Signature::one_of(
            vec![
                // 2 args - numeric, percentile (float)
                TypeSignature::Coercible(vec![
                    Coercion::new_implicit(
                        TypeSignatureClass::Float,
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Float64,
                    ),
                    Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_float64()),
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Float64,
                    ),
                ]),
                // 3 args - numeric, percentile (float), number of centroid for T-Digest (integer)
                TypeSignature::Coercible(vec![
                    Coercion::new_implicit(
                        TypeSignatureClass::Float,
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Float64,
                    ),
                    Coercion::new_implicit(
                        TypeSignatureClass::Native(logical_float64()),
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Float64,
                    ),
                    Coercion::new_implicit(
                        TypeSignatureClass::Integer,
                        vec![TypeSignatureClass::Numeric],
                        NativeType::Int64,
                    ),
                ]),
            ],
            Volatility::Immutable,
        );
        Self { signature }
    }

    pub(crate) fn create_accumulator(
        &self,
        args: &AccumulatorArgs,
    ) -> Result<ApproxPercentileAccumulator> {
        let AccumulatorParams {
            percentile,
            max_size,
            data_type,
        } = accumulator_params(args)?;

        Ok(ApproxPercentileAccumulator::new_with_max_size(
            percentile, data_type, max_size,
        ))
    }
}

/// The parameters an accumulator is built from.
struct AccumulatorParams {
    /// The percentile to estimate, with a descending ordering folded in.
    percentile: f64,
    /// The maximum number of centroids of the t-digest.
    max_size: usize,
    /// The input type, which is also the result type.
    data_type: DataType,
}

fn accumulator_params(args: &AccumulatorArgs) -> Result<AccumulatorParams> {
    let percentile = validate_percentile_expr(&args.exprs[1], "APPROX_PERCENTILE_CONT")?;

    let is_descending = args
        .order_bys
        .first()
        .map(|sort_expr| sort_expr.options.descending)
        .unwrap_or(false);

    let percentile = if is_descending {
        1.0 - percentile
    } else {
        percentile
    };

    let max_size = if args.exprs.len() == 3 {
        validate_input_max_size_expr(&args.exprs[2])?
    } else {
        DEFAULT_MAX_SIZE
    };

    let data_type = args.expr_fields[0].data_type();
    match data_type {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => {
            Ok(AccumulatorParams {
                percentile,
                max_size,
                data_type: data_type.clone(),
            })
        }
        other => {
            not_impl_err!(
                "Support for 'APPROX_PERCENTILE_CONT' for data type {other} is not implemented"
            )
        }
    }
}

fn validate_input_max_size_expr(expr: &Arc<dyn PhysicalExpr>) -> Result<usize> {
    let scalar_value = get_scalar_value(expr).map_err(|_e| {
        DataFusionError::Plan(
            "Tdigest max_size value for 'APPROX_PERCENTILE_CONT' must be a literal"
                .to_string(),
        )
    })?;

    let max_size = match scalar_value {
        ScalarValue::UInt8(Some(q)) => q as usize,
        ScalarValue::UInt16(Some(q)) => q as usize,
        ScalarValue::UInt32(Some(q)) => q as usize,
        ScalarValue::UInt64(Some(q)) => q as usize,
        ScalarValue::Int32(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int64(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int16(Some(q)) if q > 0 => q as usize,
        ScalarValue::Int8(Some(q)) if q > 0 => q as usize,
        sv => {
            return plan_err!(
                "Tdigest max_size value for 'APPROX_PERCENTILE_CONT' must be UInt > 0 literal (got data type {}).",
                sv.data_type()
            );
        }
    };

    Ok(max_size)
}

impl AggregateUDFImpl for ApproxPercentileCont {
    /// See [`TDigest::to_scalar_state()`] for a description of the serialized
    /// state.
    fn state_fields(&self, args: StateFieldsArgs) -> Result<Vec<FieldRef>> {
        Ok(vec![
            Field::new(
                format_state_name(args.name, "max_size"),
                DataType::UInt64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "sum"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "count"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "max"),
                DataType::Float64,
                false,
            ),
            Field::new(
                format_state_name(args.name, "min"),
                DataType::Float64,
                false,
            ),
            Field::new_list(
                format_state_name(args.name, "centroids"),
                Field::new_list_field(DataType::Float64, true),
                false,
            ),
        ]
        .into_iter()
        .map(Arc::new)
        .collect())
    }

    fn name(&self) -> &str {
        "approx_percentile_cont"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    #[inline]
    fn accumulator(&self, acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(self.create_accumulator(&acc_args)?))
    }

    fn groups_accumulator_supported(&self, args: AccumulatorArgs) -> bool {
        accumulator_params(&args).is_ok()
    }

    fn create_groups_accumulator(
        &self,
        args: AccumulatorArgs,
    ) -> Result<Box<dyn GroupsAccumulator>> {
        let AccumulatorParams {
            percentile,
            max_size,
            data_type,
        } = accumulator_params(&args)?;

        Ok(Box::new(ApproxPercentileGroupsAccumulator::new(
            percentile, data_type, max_size,
        )))
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if !arg_types[0].is_numeric() {
            return plan_err!("approx_percentile_cont requires numeric input types");
        }
        if arg_types.len() == 3 && !arg_types[2].is_integer() {
            return plan_err!(
                "approx_percentile_cont requires integer centroids input types"
            );
        }
        Ok(arg_types[0].clone())
    }

    fn supports_within_group_clause(&self) -> bool {
        true
    }

    fn documentation(&self) -> Option<&Documentation> {
        self.doc()
    }
}

#[derive(Debug)]
pub struct ApproxPercentileAccumulator {
    digest: TDigest,
    percentile: f64,
    return_type: DataType,
}

impl ApproxPercentileAccumulator {
    pub fn new(percentile: f64, return_type: DataType) -> Self {
        Self {
            digest: TDigest::new(DEFAULT_MAX_SIZE),
            percentile,
            return_type,
        }
    }

    pub fn new_with_max_size(
        percentile: f64,
        return_type: DataType,
        max_size: usize,
    ) -> Self {
        Self {
            digest: TDigest::new(max_size),
            percentile,
            return_type,
        }
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn max_size(&self) -> usize {
        self.digest.max_size()
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn merge_digests(&mut self, digests: &[TDigest]) {
        let digests = digests.iter().chain(std::iter::once(&self.digest));
        self.digest = TDigest::merge_digests(digests)
    }

    // pub(crate) for approx_percentile_cont_with_weight
    pub(crate) fn convert_to_float(values: &ArrayRef) -> Result<Vec<f64>> {
        debug_assert!(
            values.null_count() == 0,
            "convert_to_float assumes nulls have already been filtered out"
        );
        match values.data_type() {
            DataType::Float64 => {
                let array = downcast_value!(values, Float64Array);
                Ok(array.values().iter().copied().collect::<Vec<_>>())
            }
            DataType::Float32 => {
                let array = downcast_value!(values, Float32Array);
                Ok(array.values().iter().map(|v| *v as f64).collect::<Vec<_>>())
            }
            DataType::Float16 => {
                let array = downcast_value!(values, Float16Array);
                Ok(array
                    .values()
                    .iter()
                    .map(|v| v.to_f64())
                    .collect::<Vec<_>>())
            }
            e => internal_err!(
                "APPROX_PERCENTILE_CONT is not expected to receive the type {e:?}"
            ),
        }
    }
}

impl Accumulator for ApproxPercentileAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(self.digest.to_scalar_state().into_iter().collect())
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        // Remove any nulls before computing the percentile
        let mut values = Arc::clone(&values[0]);
        if values.null_count() > 0 {
            values = filter(&values, &is_not_null(&values)?)?;
        }
        let sorted_values = &arrow::compute::sort(&values, None)?;
        let sorted_values = ApproxPercentileAccumulator::convert_to_float(sorted_values)?;
        self.digest = self.digest.merge_sorted_f64(&sorted_values);
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.digest.count() == 0.0 {
            return ScalarValue::try_from(self.return_type.clone());
        }
        let q = self.digest.estimate_quantile(self.percentile);

        // These acceptable return types MUST match the validation in
        // ApproxPercentile::create_accumulator.
        Ok(match &self.return_type {
            DataType::Float16 => ScalarValue::Float16(Some(half::f16::from_f64(q))),
            DataType::Float32 => ScalarValue::Float32(Some(q as f32)),
            DataType::Float64 => ScalarValue::Float64(Some(q)),
            v => unreachable!("unexpected return type {}", v),
        })
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }

        let states = (0..states[0].len())
            .map(|index| {
                states
                    .iter()
                    .map(|array| ScalarValue::try_from_array(array, index))
                    .collect::<Result<Vec<_>>>()
                    .map(|state| TDigest::from_scalar_state(&state))
            })
            .collect::<Result<Vec<_>>>()?;

        self.merge_digests(&states);

        Ok(())
    }

    fn size(&self) -> usize {
        size_of_val(self) + self.digest.size() - size_of_val(&self.digest)
            + self.return_type.size()
            - size_of_val(&self.return_type)
    }
}

/// The number of values a group buffers before they are merged into its
/// digest. A merge sorts the buffer and folds it into the digest, so merging
/// a few hundred values at a time keeps the cost per value small while the
/// buffer of a group stays a few kilobytes.
const GROUP_BUFFER_LIMIT: usize = 256;

/// [`GroupsAccumulator`] for `approx_percentile_cont`: one t-digest per group.
///
/// Rows arrive interleaved across groups, often one row per group per
/// batch. Folding each batch into each group's digest costs a pass over the
/// digest's centroids per group per batch, which is what makes the generic
/// adapter over [`ApproxPercentileAccumulator`] slow when groups are spread
/// over many batches. Instead, each group buffers its values and folds them
/// into its digest only when the buffer is full and when the group is
/// emitted. A group whose values fit in one buffer builds its digest in one
/// merge, which is the same digest the row accumulator builds for it.
#[derive(Debug)]
pub struct ApproxPercentileGroupsAccumulator {
    percentile: f64,
    return_type: DataType,
    max_size: usize,
    buffer_limit: usize,
    /// Per group, the values not yet merged into the digest.
    buffers: Vec<Vec<f64>>,
    /// Per group, the digest of the merged values. `None` until the first
    /// merge, so a group that never fills its buffer allocates no digest.
    digests: Vec<Option<TDigest>>,
    /// The bytes the buffers and the digests hold. Kept current on every
    /// change so [`GroupsAccumulator::size`] costs nothing per group.
    bytes: usize,
}

impl ApproxPercentileGroupsAccumulator {
    pub fn new(percentile: f64, return_type: DataType, max_size: usize) -> Self {
        Self {
            percentile,
            return_type,
            max_size,
            buffer_limit: GROUP_BUFFER_LIMIT.max(2 * max_size),
            buffers: Vec::new(),
            digests: Vec::new(),
            bytes: 0,
        }
    }

    fn resize(&mut self, total_num_groups: usize) {
        if total_num_groups > self.buffers.len() {
            self.buffers.resize_with(total_num_groups, Vec::new);
            self.digests.resize_with(total_num_groups, || None);
        }
    }

    fn push(&mut self, group: usize, value: f64) {
        self.buffers[group].push(value);
        self.bytes += size_of::<f64>();

        if self.buffers[group].len() >= self.buffer_limit {
            self.flush(group);
        }
    }

    /// Fold the buffered values of a group into its digest.
    fn flush(&mut self, group: usize) {
        let values = std::mem::take(&mut self.buffers[group]);
        if values.is_empty() {
            return;
        }
        self.bytes -= values.len() * size_of::<f64>();

        let digest = match self.digests[group].take() {
            Some(digest) => {
                self.bytes -= digest.size();
                digest.merge_unsorted_f64(values)
            }
            None => TDigest::new(self.max_size).merge_unsorted_f64(values),
        };

        self.bytes += digest.size();
        self.digests[group] = Some(digest);
    }

    /// Fold a digest into a group's digest.
    fn merge(&mut self, group: usize, incoming: TDigest) {
        self.flush(group);

        let digest = match self.digests[group].take() {
            Some(digest) => {
                self.bytes -= digest.size();
                TDigest::merge_digests([&digest, &incoming])
            }
            None => incoming,
        };

        self.bytes += digest.size();
        self.digests[group] = Some(digest);
    }

    /// Remove the digests of the groups to emit, with their buffers folded
    /// in. A group without values yields `None`.
    fn take_digests(&mut self, emit_to: EmitTo) -> Vec<Option<TDigest>> {
        let count = match emit_to {
            EmitTo::All => self.buffers.len(),
            EmitTo::First(n) => n.min(self.buffers.len()),
        };
        for group in 0..count {
            self.flush(group);
        }

        let digests = emit_to.take_needed(&mut self.digests);
        let buffers = emit_to.take_needed(&mut self.buffers);
        debug_assert!(buffers.iter().all(Vec::is_empty));

        self.bytes = self
            .buffers
            .iter()
            .map(|b| b.len() * size_of::<f64>())
            .sum::<usize>()
            + self
                .digests
                .iter()
                .flatten()
                .map(TDigest::size)
                .sum::<usize>();

        digests
            .into_iter()
            .map(|digest| digest.filter(|digest| digest.count() > 0.0))
            .collect()
    }
}

/// Builders for the six state columns, laid out as
/// [`TDigest::to_scalar_state`] lays them out.
struct StateBuilders {
    max_size: UInt64Builder,
    sum: Float64Builder,
    count: Float64Builder,
    max: Float64Builder,
    min: Float64Builder,
    centroids: ListBuilder<Float64Builder>,
}

impl StateBuilders {
    fn with_capacity(rows: usize) -> Self {
        Self {
            max_size: UInt64Builder::with_capacity(rows),
            sum: Float64Builder::with_capacity(rows),
            count: Float64Builder::with_capacity(rows),
            max: Float64Builder::with_capacity(rows),
            min: Float64Builder::with_capacity(rows),
            centroids: ListBuilder::with_capacity(Float64Builder::new(), rows),
        }
    }

    fn append(&mut self, digest: &TDigest) {
        self.max_size.append_value(digest.max_size() as u64);
        self.sum.append_value(digest.sum());
        self.count.append_value(digest.count());
        self.max.append_value(digest.max());
        self.min.append_value(digest.min());
        for centroid in digest.centroids() {
            self.centroids.values().append_value(centroid.mean());
            self.centroids.values().append_value(centroid.weight());
        }
        self.centroids.append(true);
    }

    fn finish(mut self) -> Vec<ArrayRef> {
        vec![
            Arc::new(self.max_size.finish()),
            Arc::new(self.sum.finish()),
            Arc::new(self.count.finish()),
            Arc::new(self.max.finish()),
            Arc::new(self.min.finish()),
            Arc::new(self.centroids.finish()),
        ]
    }
}

impl GroupsAccumulator for ApproxPercentileGroupsAccumulator {
    fn update_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        // The percentile and the centroid count arrive as constant columns
        // after the value column, as for the row accumulator.
        let Some(values) = values.first() else {
            return internal_err!("APPROX_PERCENTILE_CONT received no arguments");
        };
        self.resize(total_num_groups);
        match values.data_type() {
            DataType::Float64 => accumulate(
                group_indices,
                values.as_primitive::<Float64Type>(),
                opt_filter,
                |group, value| self.push(group, value),
            ),
            DataType::Float32 => accumulate(
                group_indices,
                values.as_primitive::<Float32Type>(),
                opt_filter,
                |group, value| self.push(group, value as f64),
            ),
            DataType::Float16 => accumulate(
                group_indices,
                values.as_primitive::<Float16Type>(),
                opt_filter,
                |group, value| self.push(group, value.to_f64()),
            ),
            other => {
                return internal_err!(
                    "APPROX_PERCENTILE_CONT is not expected to receive the type {other:?}"
                );
            }
        }

        Ok(())
    }

    fn evaluate(&mut self, emit_to: EmitTo) -> Result<ArrayRef> {
        let digests = self.take_digests(emit_to);
        let quantiles = digests.iter().map(|digest| {
            digest
                .as_ref()
                .map(|d| d.estimate_quantile(self.percentile))
        });

        // These return types MUST match the validation in `accumulator_params`,
        // and the conversions MUST match `ApproxPercentileAccumulator::evaluate`.
        Ok(match &self.return_type {
            DataType::Float16 => Arc::new(Float16Array::from_iter(
                quantiles.map(|q| q.map(half::f16::from_f64)),
            )),
            DataType::Float32 => Arc::new(Float32Array::from_iter(
                quantiles.map(|q| q.map(|q| q as f32)),
            )),
            DataType::Float64 => Arc::new(Float64Array::from_iter(quantiles)),
            other => return internal_err!("unexpected return type {other}"),
        })
    }

    fn state(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        let digests = self.take_digests(emit_to);
        let empty = TDigest::new(self.max_size);

        let mut builders = StateBuilders::with_capacity(digests.len());
        for digest in &digests {
            builders.append(digest.as_ref().unwrap_or(&empty));
        }

        Ok(builders.finish())
    }

    fn merge_batch(
        &mut self,
        values: &[ArrayRef],
        group_indices: &[usize],
        // The filter applies in the partial phase; the final phase has none.
        _opt_filter: Option<&BooleanArray>,
        total_num_groups: usize,
    ) -> Result<()> {
        if values.len() != 6 {
            return internal_err!(
                "APPROX_PERCENTILE_CONT expects 6 state arrays, got {}",
                values.len()
            );
        }
        self.resize(total_num_groups);

        let max_sizes = values[0].as_primitive::<UInt64Type>();
        let sums = values[1].as_primitive::<Float64Type>();
        let counts = values[2].as_primitive::<Float64Type>();
        let maxes = values[3].as_primitive::<Float64Type>();
        let mins = values[4].as_primitive::<Float64Type>();
        let centroids = values[5].as_list::<i32>();
        let means_and_weights = centroids.values().as_primitive::<Float64Type>().values();
        let offsets = centroids.value_offsets();

        for (row, &group) in group_indices.iter().enumerate() {
            // A group without values has an empty state, which merges as nothing.
            if counts.value(row) == 0.0 {
                continue;
            }

            let (start, end) = (offsets[row] as usize, offsets[row + 1] as usize);
            let centroids = means_and_weights[start..end]
                .chunks(2)
                .map(|pair| Centroid::new(pair[0], pair[1]))
                .collect();

            let incoming = TDigest::from_parts(
                max_sizes.value(row) as usize,
                sums.value(row),
                counts.value(row),
                maxes.value(row),
                mins.value(row),
                centroids,
            );

            self.merge(group, incoming);
        }

        Ok(())
    }

    fn convert_to_state(
        &self,
        values: &[ArrayRef],
        opt_filter: Option<&BooleanArray>,
    ) -> Result<Vec<ArrayRef>> {
        let Some(values) = values.first() else {
            return internal_err!("APPROX_PERCENTILE_CONT received no arguments");
        };

        // Each row becomes the state of a digest of that one value. A null or
        // filtered row becomes the state of an empty digest, which the merge
        // treats as nothing; the state columns are not nullable.
        let valid = filtered_null_mask(opt_filter, values.as_ref());
        let empty = TDigest::new(self.max_size);
        let mut builders = StateBuilders::with_capacity(values.len());

        let mut append = |row: usize, value: f64| {
            if valid.as_ref().is_none_or(|valid| valid.is_valid(row)) {
                let single =
                    TDigest::new_with_centroid(self.max_size, Centroid::new(value, 1.0));
                builders.append(&single);
            } else {
                builders.append(&empty);
            }
        };

        match values.data_type() {
            DataType::Float64 => {
                let values = values.as_primitive::<Float64Type>();
                for (row, &value) in values.values().iter().enumerate() {
                    append(row, value);
                }
            }
            DataType::Float32 => {
                let values = values.as_primitive::<Float32Type>();
                for (row, &value) in values.values().iter().enumerate() {
                    append(row, value as f64);
                }
            }
            DataType::Float16 => {
                let values = values.as_primitive::<Float16Type>();
                for (row, &value) in values.values().iter().enumerate() {
                    append(row, value.to_f64());
                }
            }
            other => {
                return internal_err!(
                    "APPROX_PERCENTILE_CONT is not expected to receive the type {other:?}"
                );
            }
        }

        Ok(builders.finish())
    }

    fn supports_convert_to_state(&self) -> bool {
        true
    }

    fn size(&self) -> usize {
        size_of_val(self)
            + self.bytes
            + self.buffers.capacity() * size_of::<Vec<f64>>()
            + self.digests.capacity() * size_of::<Option<TDigest>>()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array};
    use arrow::datatypes::{DataType, Float32Type, Float64Type};

    use datafusion_common::ScalarValue;
    use datafusion_expr::{Accumulator, EmitTo, GroupsAccumulator};
    use datafusion_functions_aggregate_common::tdigest::TDigest;

    use crate::approx_percentile_cont::{
        ApproxPercentileAccumulator, ApproxPercentileGroupsAccumulator,
    };

    /// `count` values in `0..1000`, in a scrambled order that depends on `seed`.
    fn scrambled(seed: u64, count: usize) -> Vec<f64> {
        (0..count as u64)
            .map(|i| ((i * 7919 + seed * 104_729) % 1000) as f64)
            .collect()
    }

    /// Interleave the groups' values row by row, as a hash aggregate sees
    /// them: `[g0, g1, g2, g0, g1, g2, ...]`.
    fn interleave(groups: &[Vec<f64>]) -> (Vec<f64>, Vec<usize>) {
        let longest = groups.iter().map(Vec::len).max().unwrap_or(0);
        let mut values = Vec::new();
        let mut indices = Vec::new();
        for row in 0..longest {
            for (group, group_values) in groups.iter().enumerate() {
                if let Some(value) = group_values.get(row) {
                    values.push(*value);
                    indices.push(group);
                }
            }
        }
        (values, indices)
    }

    /// Feed the interleaved rows in batches of `batch_rows`.
    fn update_in_batches(
        acc: &mut ApproxPercentileGroupsAccumulator,
        values: &[f64],
        indices: &[usize],
        batch_rows: usize,
        total_num_groups: usize,
    ) {
        for (values, indices) in values.chunks(batch_rows).zip(indices.chunks(batch_rows))
        {
            let array: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
            acc.update_batch(&[array], indices, None, total_num_groups)
                .unwrap();
        }
    }

    /// What the row accumulator answers for one group.
    fn row_estimate(percentile: f64, values: &[f64]) -> Option<f64> {
        let mut acc = ApproxPercentileAccumulator::new(percentile, DataType::Float64);
        let array: ArrayRef = Arc::new(Float64Array::from(values.to_vec()));
        acc.update_batch(&[array]).unwrap();
        match acc.evaluate().unwrap() {
            ScalarValue::Float64(value) => value,
            other => panic!("unexpected result {other:?}"),
        }
    }

    fn f64_results(array: &ArrayRef) -> Vec<Option<f64>> {
        array.as_primitive::<Float64Type>().iter().collect()
    }

    #[test]
    fn test_combine_approx_percentile_accumulator() {
        let mut digests: Vec<TDigest> = Vec::new();

        // one TDigest with 50_000 values from 1 to 1_000
        for _ in 1..=50 {
            let t = TDigest::new(100);
            let values: Vec<_> = (1..=1_000).map(f64::from).collect();
            let t = t.merge_unsorted_f64(values);
            digests.push(t)
        }

        let t1 = TDigest::merge_digests(&digests);
        let t2 = TDigest::merge_digests(&digests);

        let mut accumulator =
            ApproxPercentileAccumulator::new_with_max_size(0.5, DataType::Float64, 100);

        accumulator.merge_digests(&[t1]);
        assert_eq!(accumulator.digest.count(), 50_000.0);
        accumulator.merge_digests(&[t2]);
        assert_eq!(accumulator.digest.count(), 100_000.0);
    }

    #[test]
    fn test_groups_accumulator_matches_row_accumulator_for_small_groups() {
        // Groups with fewer values than centroids keep every value as its own
        // centroid, so the batching cannot change the estimate: the two
        // accumulators must agree exactly.
        let groups = vec![
            scrambled(1, 30),
            scrambled(2, 45),
            scrambled(3, 1),
            scrambled(4, 99),
        ];
        let (values, indices) = interleave(&groups);

        let mut acc = ApproxPercentileGroupsAccumulator::new(0.9, DataType::Float64, 100);
        update_in_batches(&mut acc, &values, &indices, 7, groups.len());
        let results = f64_results(&acc.evaluate(EmitTo::All).unwrap());

        let expected: Vec<_> = groups.iter().map(|g| row_estimate(0.9, g)).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_groups_accumulator_state_round_trip() {
        // Two partitions each see half of the rows and emit their state; the
        // final phase merges both. Every group stays under the centroid
        // limit, so the merged digest holds the same points as the row
        // accumulator's and the estimates must agree exactly.
        let groups = vec![scrambled(5, 40), scrambled(6, 20)];
        let (values, indices) = interleave(&groups);
        let half = values.len() / 2;

        let mut states = Vec::new();
        for (values, indices) in [
            (&values[..half], &indices[..half]),
            (&values[half..], &indices[half..]),
        ] {
            let mut partial =
                ApproxPercentileGroupsAccumulator::new(0.25, DataType::Float64, 100);
            update_in_batches(&mut partial, values, indices, 8, groups.len());
            let state = partial.state(EmitTo::All).unwrap();
            assert_eq!(state.len(), 6);
            assert_eq!(state[0].len(), groups.len());
            states.push(state);
        }

        let mut final_ =
            ApproxPercentileGroupsAccumulator::new(0.25, DataType::Float64, 100);
        for state in &states {
            final_.merge_batch(state, &[0, 1], None, 2).unwrap();
        }
        let results = f64_results(&final_.evaluate(EmitTo::All).unwrap());

        let expected: Vec<_> = groups.iter().map(|g| row_estimate(0.25, g)).collect();
        assert_eq!(results, expected);
    }

    #[test]
    fn test_groups_accumulator_folds_full_buffers() {
        let values = scrambled(7, 3_000);
        let indices = vec![0; values.len()];

        let mut acc = ApproxPercentileGroupsAccumulator::new(0.9, DataType::Float64, 100);
        update_in_batches(&mut acc, &values, &indices, 512, 1);

        // The buffer was folded into a digest on the way, and the digest
        // carries the count of everything folded so far.
        let digest = acc.digests[0].as_ref().expect("a digest after 3000 values");
        assert!(acc.buffers[0].len() < acc.buffer_limit);
        assert_eq!(digest.count() + acc.buffers[0].len() as f64, 3_000.0);

        // The values are uniform over 0..1000, so p90 is near 900. Both
        // accumulators are approximate; the estimates must be close to the
        // truth and to each other.
        let result = f64_results(&acc.evaluate(EmitTo::All).unwrap())[0].unwrap();
        let expected = row_estimate(0.9, &values).unwrap();
        assert!((result - 899.0).abs() < 15.0, "p90 estimate {result}");
        assert!((result - expected).abs() < 15.0, "{result} vs {expected}");
    }

    #[test]
    fn test_groups_accumulator_skips_nulls_filtered_rows_and_empty_groups() {
        let values: ArrayRef = Arc::new(Float64Array::from(vec![
            Some(1.0),
            None,
            Some(3.0),
            Some(100.0),
        ]));
        let filter = BooleanArray::from(vec![true, true, true, false]);

        let mut acc = ApproxPercentileGroupsAccumulator::new(0.5, DataType::Float64, 100);
        acc.update_batch(&[values], &[0, 0, 0, 1], Some(&filter), 3)
            .unwrap();

        // Group 1 only had a filtered row and group 2 never had one.
        let results = f64_results(&acc.evaluate(EmitTo::All).unwrap());
        assert_eq!(results, vec![row_estimate(0.5, &[1.0, 3.0]), None, None]);
    }

    #[test]
    fn test_groups_accumulator_empty_state_merges_as_nothing() {
        let mut acc = ApproxPercentileGroupsAccumulator::new(0.5, DataType::Float64, 100);
        acc.update_batch(
            &[Arc::new(Float64Array::from(vec![2.0, 4.0]))],
            &[0, 0],
            None,
            2,
        )
        .unwrap();

        // Group 1 has no values: its state row is an empty digest.
        let state = acc.state(EmitTo::All).unwrap();
        assert_eq!(state[2].as_primitive::<Float64Type>().value(1), 0.0);

        let mut merged =
            ApproxPercentileGroupsAccumulator::new(0.5, DataType::Float64, 100);
        merged.merge_batch(&state, &[0, 1], None, 2).unwrap();
        let results = f64_results(&merged.evaluate(EmitTo::All).unwrap());
        assert_eq!(results, vec![row_estimate(0.5, &[2.0, 4.0]), None]);
    }

    #[test]
    fn test_convert_to_state_merges_like_update() {
        let groups = vec![scrambled(8, 12), scrambled(9, 9)];
        let (values, indices) = interleave(&groups);
        let array: ArrayRef = Arc::new(Float64Array::from(values.clone()));

        // Every other row is filtered out of both paths.
        let filter =
            BooleanArray::from((0..values.len()).map(|i| i % 2 == 0).collect::<Vec<_>>());

        let mut updated =
            ApproxPercentileGroupsAccumulator::new(0.75, DataType::Float64, 100);
        updated
            .update_batch(&[Arc::clone(&array)], &indices, Some(&filter), 2)
            .unwrap();

        let converter =
            ApproxPercentileGroupsAccumulator::new(0.75, DataType::Float64, 100);
        assert!(converter.supports_convert_to_state());
        let state = converter.convert_to_state(&[array], Some(&filter)).unwrap();
        assert_eq!(state[0].len(), values.len());

        let mut merged =
            ApproxPercentileGroupsAccumulator::new(0.75, DataType::Float64, 100);
        merged.merge_batch(&state, &indices, None, 2).unwrap();

        assert_eq!(
            f64_results(&merged.evaluate(EmitTo::All).unwrap()),
            f64_results(&updated.evaluate(EmitTo::All).unwrap())
        );
    }

    #[test]
    fn test_emit_first_shifts_the_remaining_groups() {
        let first = scrambled(10, 25);
        let second = scrambled(11, 25);
        let more = scrambled(12, 25);
        let (values, indices) = interleave(&[first.clone(), second.clone()]);

        let mut acc = ApproxPercentileGroupsAccumulator::new(0.5, DataType::Float64, 100);
        update_in_batches(&mut acc, &values, &indices, 10, 2);

        let emitted = f64_results(&acc.evaluate(EmitTo::First(1)).unwrap());
        assert_eq!(emitted, vec![row_estimate(0.5, &first)]);

        // The second group is now group 0 and keeps receiving values.
        update_in_batches(&mut acc, &more, &vec![0; more.len()], 10, 1);
        let rest = f64_results(&acc.evaluate(EmitTo::All).unwrap());
        let combined: Vec<f64> = second.iter().chain(more.iter()).copied().collect();
        assert_eq!(rest, vec![row_estimate(0.5, &combined)]);
    }

    #[test]
    fn test_groups_accumulator_keeps_the_input_type() {
        let values: ArrayRef = Arc::new(Float32Array::from(vec![1.5f32, 2.5, 10.0]));

        let mut acc = ApproxPercentileGroupsAccumulator::new(0.5, DataType::Float32, 100);
        acc.update_batch(&[values], &[0, 0, 0], None, 1).unwrap();
        let result = acc.evaluate(EmitTo::All).unwrap();
        assert_eq!(result.data_type(), &DataType::Float32);

        let mut row = ApproxPercentileAccumulator::new(0.5, DataType::Float32);
        row.update_batch(&[Arc::new(Float32Array::from(vec![1.5f32, 2.5, 10.0]))])
            .unwrap();
        let expected = match row.evaluate().unwrap() {
            ScalarValue::Float32(value) => value,
            other => panic!("unexpected result {other:?}"),
        };
        assert_eq!(
            result.as_primitive::<Float32Type>().iter().next().unwrap(),
            expected
        );
    }
}
