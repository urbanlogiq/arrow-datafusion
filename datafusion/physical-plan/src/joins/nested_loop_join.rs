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

//! [`NestedLoopJoinExec`]: joins without equijoin (equality predicates).

use std::fmt::Formatter;
use std::ops::{BitOr, ControlFlow};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::task::Poll;

use super::utils::{
    asymmetric_join_output_partitioning, build_batch_offsets, flat_index_to_batch_row,
    need_produce_result_in_final, reorder_output_after_swap, swap_join_projection,
};
use crate::common::can_project;
use crate::execution_plan::{EmissionType, boundedness_from_children};
use crate::joins::SharedBitmapBuilder;
use crate::joins::utils::{
    BuildProbeJoinMetrics, ColumnIndex, JoinFilter, OnceAsync, OnceFut,
    build_join_schema, check_join_is_valid, estimate_join_statistics,
    need_produce_right_in_final,
};
use crate::metrics::{
    Count, ExecutionPlanMetricsSet, MetricBuilder, MetricType, MetricsSet, RatioMetrics,
};
use crate::projection::{
    EmbeddedProjection, JoinData, ProjectionExec, try_embed_projection,
    try_pushdown_through_join,
};
use crate::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, RecordBatchStream, SendableRecordBatchStream,
    check_if_same_properties,
};

use arrow::array::{
    Array, AsArray, BooleanArray, BooleanBufferBuilder, RecordBatchOptions, UInt32Array,
    UInt64Array, new_null_array,
};
use arrow::buffer::BooleanBuffer;
use arrow::compute::{
    BatchCoalescer, concat, concat_batches, filter, filter_record_batch,
    interleave_record_batch, not, take, take_record_batch,
};
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_schema::DataType;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{
    JoinSide, Result, ScalarValue, Statistics, assert_eq_or_internal_err,
    internal_datafusion_err, internal_err, project_schema, unwrap_or_internal_err,
};
use datafusion_execution::TaskContext;
use datafusion_execution::disk_manager::RefCountedTempFile;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::JoinType;
use datafusion_physical_expr::equivalence::{
    ProjectionMapping, join_equivalence_properties,
};

use datafusion_physical_expr::projection::{ProjectionRef, combine_projections};
use futures::{Stream, StreamExt, TryStreamExt};
use log::debug;
use parking_lot::Mutex;

use crate::metrics::SpillMetrics;
use crate::spill::replayable_spill_input::ReplayableStreamSource;
use crate::spill::spill_manager::SpillManager;

#[expect(rustdoc::private_intra_doc_links)]
/// NestedLoopJoinExec is a build-probe join operator designed for joins that
/// do not have equijoin keys in their `ON` clause.
///
/// # Execution Flow
///
/// ```text
///                                                Incoming right batch
///                Left Side Buffered Batches
///                       ┌───────────┐              ┌───────────────┐
///                       │ ┌───────┐ │              │               │
///                       │ │       │ │              │               │
///  Current Left Row ───▶│ ├───────├─┤──────────┐   │               │
///                       │ │       │ │          │   └───────────────┘
///                       │ │       │ │          │           │
///                       │ │       │ │          │           │
///                       │ └───────┘ │          │           │
///                       │ ┌───────┐ │          │           │
///                       │ │       │ │          │     ┌─────┘
///                       │ │       │ │          │     │
///                       │ │       │ │          │     │
///                       │ │       │ │          │     │
///                       │ │       │ │          │     │
///                       │ └───────┘ │          ▼     ▼
///                       │   ......  │  ┌──────────────────────┐
///                       │           │  │X (Cartesian Product) │
///                       │           │  └──────────┬───────────┘
///                       └───────────┘             │
///                                                 │
///                                                 ▼
///                                      ┌───────┬───────────────┐
///                                      │       │               │
///                                      │       │               │
///                                      │       │               │
///                                      └───────┴───────────────┘
///                                        Intermediate Batch
///                                  (For join predicate evaluation)
/// ```
///
/// The execution follows a two-phase design:
///
/// ## 1. Buffering Left Input
/// - The operator eagerly buffers all left-side input batches into memory,
///   util a memory limit is reached.
///   Currently, an out-of-memory error will be thrown if all the left-side input batches
///   cannot fit into memory at once.
///   In the future, it's possible to make this case finish execution. (see
///   'Memory-limited Execution' section)
/// - The rationale for buffering the left side is that scanning the right side
///   can be expensive (e.g., decoding Parquet files), so buffering more left
///   rows reduces the number of right-side scan passes required.
///
/// ## 2. Probing Right Input
/// - Right-side input is streamed batch by batch.
/// - For each right-side batch:
///   - It evaluates the join filter against the full buffered left input.
///     This results in a Cartesian product between the right batch and each
///     left row -- with the join predicate/filter applied -- for each inner
///     loop iteration.
///   - Matched results are accumulated into an output buffer. (see more in
///     `Output Buffering Strategy` section)
/// - This process continues until all right-side input is consumed.
///
/// # Producing unmatched build-side data
/// - For special join types like left/full joins, it's required to also output
///   unmatched pairs. During execution, bitmaps are kept for both left and right
///   sides of the input; they'll be handled by dedicated states in `NLJStream`.
/// - The final output of the left side unmatched rows is handled by a single
///   partition for simplicity, since it only counts a small portion of the
///   execution time. (e.g. if probe side has 10k rows, the final output of
///   unmatched build side only roughly counts for 1/10k of the total time)
///
/// # Output Buffering Strategy
/// The operator uses an intermediate output buffer to accumulate results. Once
/// the output threshold is reached (currently set to the same value as
/// `batch_size` in the configuration), the results will be eagerly output.
///
/// # Extra Notes
/// - The operator always considers the **left** side as the build (buffered) side.
///   Therefore, the physical optimizer should assign the smaller input to the left.
/// - The design try to minimize the intermediate data size to approximately
///   1 batch, for better cache locality and memory efficiency.
///
/// # Memory-limited Execution
/// When the memory budget is exceeded during left-side buffering, the operator
/// falls back to a multi-pass strategy:
/// 1. Buffer as many left rows as fit in memory (one "chunk")
/// 2. On the first pass, the right side is both processed and spilled to disk
/// 3. For each subsequent left chunk, the right side is re-read from the spill file
///
/// The fallback is triggered automatically when the initial in-memory load
/// fails with `ResourcesExhausted` and disk spilling is available. Each
/// output partition independently re-executes the left child and manages
/// its own spill state.
///
/// All join types are supported. For RIGHT/FULL/RIGHT SEMI/RIGHT ANTI/
/// RIGHT MARK joins, a global right-side bitmap (indexed by right batch
/// sequence number) accumulates matches across all left chunks. After the
/// last left chunk is processed, the right side is replayed one more time
/// to emit unmatched right rows using the accumulated bitmap.
///
/// Tracking issue: <https://github.com/apache/datafusion/issues/15760>
///
/// # Clone / Shared State
/// Note this structure includes a [`OnceAsync`] that is used to coordinate the
/// loading of the left side with the processing in each output stream.
/// Therefore it can not be [`Clone`]
#[derive(Debug)]
pub struct NestedLoopJoinExec {
    /// left side
    pub(crate) left: Arc<dyn ExecutionPlan>,
    /// right side
    pub(crate) right: Arc<dyn ExecutionPlan>,
    /// Filters which are applied while finding matching rows
    pub(crate) filter: Option<JoinFilter>,
    /// How the join is performed
    pub(crate) join_type: JoinType,
    /// The full concatenated schema of left and right children should be distinct from
    /// the output schema of the operator
    join_schema: SchemaRef,
    /// Future that consumes left input and buffers it in memory
    ///
    /// This structure is *shared* across all output streams.
    ///
    /// Each output stream waits on the `OnceAsync` to signal the completion of
    /// the build(left) side data, and buffer them all for later joining.
    build_side_data: OnceAsync<JoinLeftData>,
    /// Shared left-side spill data for OOM fallback.
    ///
    /// When `build_side_data` fails with OOM, the first partition to
    /// initiate fallback spills the entire left side to disk. Other
    /// partitions share the same spill file via this `OnceAsync`,
    /// avoiding redundant re-execution of the left child.
    left_spill_data: Arc<OnceAsync<LeftSpillData>>,
    /// Information of index and left / right placement of columns
    column_indices: Vec<ColumnIndex>,
    /// Projection to apply to the output of the join
    projection: Option<ProjectionRef>,

    /// Execution metrics
    metrics: ExecutionPlanMetricsSet,
    /// Cache holding plan properties like equivalences, output partitioning etc.
    cache: Arc<PlanProperties>,
}

/// Helps to build [`NestedLoopJoinExec`].
pub struct NestedLoopJoinExecBuilder {
    left: Arc<dyn ExecutionPlan>,
    right: Arc<dyn ExecutionPlan>,
    join_type: JoinType,
    filter: Option<JoinFilter>,
    projection: Option<ProjectionRef>,
}

impl NestedLoopJoinExecBuilder {
    /// Make a new [`NestedLoopJoinExecBuilder`].
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
    ) -> Self {
        Self {
            left,
            right,
            join_type,
            filter: None,
            projection: None,
        }
    }

    /// Set projection from the vector.
    pub fn with_projection(self, projection: Option<Vec<usize>>) -> Self {
        self.with_projection_ref(projection.map(Into::into))
    }

    /// Set projection from the shared reference.
    pub fn with_projection_ref(mut self, projection: Option<ProjectionRef>) -> Self {
        self.projection = projection;
        self
    }

    /// Set optional filter.
    pub fn with_filter(mut self, filter: Option<JoinFilter>) -> Self {
        self.filter = filter;
        self
    }

    /// Build resulting execution plan.
    pub fn build(self) -> Result<NestedLoopJoinExec> {
        let Self {
            left,
            right,
            join_type,
            filter,
            projection,
        } = self;

        let left_schema = left.schema();
        let right_schema = right.schema();
        check_join_is_valid(&left_schema, &right_schema, &[])?;
        let (join_schema, column_indices) =
            build_join_schema(&left_schema, &right_schema, &join_type);
        let join_schema = Arc::new(join_schema);
        let cache = NestedLoopJoinExec::compute_properties(
            &left,
            &right,
            &join_schema,
            join_type,
            projection.as_deref(),
        )?;
        Ok(NestedLoopJoinExec {
            left,
            right,
            filter,
            join_type,
            join_schema,
            build_side_data: Default::default(),
            left_spill_data: Arc::new(OnceAsync::default()),
            column_indices,
            projection,
            metrics: Default::default(),
            cache: Arc::new(cache),
        })
    }
}

impl From<&NestedLoopJoinExec> for NestedLoopJoinExecBuilder {
    fn from(exec: &NestedLoopJoinExec) -> Self {
        Self {
            left: Arc::clone(exec.left()),
            right: Arc::clone(exec.right()),
            join_type: exec.join_type,
            filter: exec.filter.clone(),
            projection: exec.projection.clone(),
        }
    }
}

impl NestedLoopJoinExec {
    /// Try to create a new [`NestedLoopJoinExec`]
    pub fn try_new(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        filter: Option<JoinFilter>,
        join_type: &JoinType,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        NestedLoopJoinExecBuilder::new(left, right, *join_type)
            .with_projection(projection)
            .with_filter(filter)
            .build()
    }

    /// left side
    pub fn left(&self) -> &Arc<dyn ExecutionPlan> {
        &self.left
    }

    /// right side
    pub fn right(&self) -> &Arc<dyn ExecutionPlan> {
        &self.right
    }

    /// Filters applied before join output
    pub fn filter(&self) -> Option<&JoinFilter> {
        self.filter.as_ref()
    }

    /// How the join is performed
    pub fn join_type(&self) -> &JoinType {
        &self.join_type
    }

    pub fn projection(&self) -> &Option<ProjectionRef> {
        &self.projection
    }

    /// This function creates the cache object that stores the plan properties such as schema, equivalence properties, ordering, partitioning, etc.
    fn compute_properties(
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        schema: &SchemaRef,
        join_type: JoinType,
        projection: Option<&[usize]>,
    ) -> Result<PlanProperties> {
        // Calculate equivalence properties:
        let mut eq_properties = join_equivalence_properties(
            left.equivalence_properties().clone(),
            right.equivalence_properties().clone(),
            &join_type,
            Arc::clone(schema),
            &Self::maintains_input_order(join_type),
            None,
            // No on columns in nested loop join
            &[],
        )?;

        let mut output_partitioning =
            asymmetric_join_output_partitioning(left, right, &join_type)?;

        let emission_type = if left.boundedness().is_unbounded() {
            EmissionType::Final
        } else if right.pipeline_behavior() == EmissionType::Incremental {
            match join_type {
                // If we only need to generate matched rows from the probe side,
                // we can emit rows incrementally.
                JoinType::Inner
                | JoinType::LeftSemi
                | JoinType::RightSemi
                | JoinType::Right
                | JoinType::RightAnti
                | JoinType::RightMark => EmissionType::Incremental,
                // If we need to generate unmatched rows from the *build side*,
                // we need to emit them at the end.
                JoinType::Left
                | JoinType::LeftAnti
                | JoinType::LeftMark
                | JoinType::Full => EmissionType::Both,
            }
        } else {
            right.pipeline_behavior()
        };

        if let Some(projection) = projection {
            // construct a map from the input expressions to the output expression of the Projection
            let projection_mapping = ProjectionMapping::from_indices(projection, schema)?;
            let out_schema = project_schema(schema, Some(&projection))?;
            output_partitioning =
                output_partitioning.project(&projection_mapping, &eq_properties);
            eq_properties = eq_properties.project(&projection_mapping, out_schema);
        }

        Ok(PlanProperties::new(
            eq_properties,
            output_partitioning,
            emission_type,
            boundedness_from_children([left, right]),
        ))
    }

    /// This join implementation does not preserve the input order of either side.
    fn maintains_input_order(_join_type: JoinType) -> Vec<bool> {
        vec![false, false]
    }

    pub fn contains_projection(&self) -> bool {
        self.projection.is_some()
    }

    pub fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        let projection = projection.map(Into::into);
        // check if the projection is valid
        can_project(&self.schema(), projection.as_deref())?;
        let projection =
            combine_projections(projection.as_ref(), self.projection.as_ref())?;
        NestedLoopJoinExecBuilder::from(self)
            .with_projection_ref(projection)
            .build()
    }

    /// Returns a new `ExecutionPlan` that runs NestedLoopsJoins with the left
    /// and right inputs swapped.
    ///
    /// # Notes:
    ///
    /// This function should be called BEFORE inserting any repartitioning
    /// operators on the join's children. Check [`super::HashJoinExec::swap_inputs`]
    /// for more details.
    pub fn swap_inputs(&self) -> Result<Arc<dyn ExecutionPlan>> {
        let left = self.left();
        let right = self.right();
        let new_join = NestedLoopJoinExec::try_new(
            Arc::clone(right),
            Arc::clone(left),
            self.filter().map(JoinFilter::swap),
            &self.join_type().swap(),
            swap_join_projection(
                left.schema().fields().len(),
                right.schema().fields().len(),
                self.projection.as_deref(),
                self.join_type(),
            ),
        )?;

        // For Semi/Anti joins, swap result will produce same output schema,
        // no need to wrap them into additional projection
        let plan: Arc<dyn ExecutionPlan> = if matches!(
            self.join_type(),
            JoinType::LeftSemi
                | JoinType::RightSemi
                | JoinType::LeftAnti
                | JoinType::RightAnti
                | JoinType::LeftMark
                | JoinType::RightMark
        ) || self.projection.is_some()
        {
            Arc::new(new_join)
        } else {
            reorder_output_after_swap(
                Arc::new(new_join),
                &self.left().schema(),
                &self.right().schema(),
            )?
        };

        Ok(plan)
    }

    fn with_new_children_and_same_properties(
        &self,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Self {
        let left = children.swap_remove(0);
        let right = children.swap_remove(0);

        Self {
            left,
            right,
            metrics: ExecutionPlanMetricsSet::new(),
            build_side_data: Default::default(),
            left_spill_data: Arc::new(OnceAsync::default()),
            cache: Arc::clone(&self.cache),
            filter: self.filter.clone(),
            join_type: self.join_type,
            join_schema: Arc::clone(&self.join_schema),
            column_indices: self.column_indices.clone(),
            projection: self.projection.clone(),
        }
    }
}

impl DisplayAs for NestedLoopJoinExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                let display_filter = self.filter.as_ref().map_or_else(
                    || "".to_string(),
                    |f| format!(", filter={}", f.expression()),
                );
                let display_projections = if self.contains_projection() {
                    format!(
                        ", projection=[{}]",
                        self.projection
                            .as_ref()
                            .unwrap()
                            .iter()
                            .map(|index| format!(
                                "{}@{}",
                                self.join_schema.fields().get(*index).unwrap().name(),
                                index
                            ))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    "".to_string()
                };
                write!(
                    f,
                    "NestedLoopJoinExec: join_type={:?}{}{}",
                    self.join_type, display_filter, display_projections
                )
            }
            DisplayFormatType::TreeRender => {
                if *self.join_type() != JoinType::Inner {
                    writeln!(f, "join_type={:?}", self.join_type)
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl ExecutionPlan for NestedLoopJoinExec {
    fn name(&self) -> &'static str {
        "NestedLoopJoinExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![
            Distribution::SinglePartition,
            Distribution::UnspecifiedDistribution,
        ]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        Self::maintains_input_order(self.join_type)
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left, &self.right]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        check_if_same_properties!(self, children);
        Ok(Arc::new(
            NestedLoopJoinExecBuilder::new(
                Arc::clone(&children[0]),
                Arc::clone(&children[1]),
                self.join_type,
            )
            .with_filter(self.filter.clone())
            .with_projection_ref(self.projection.clone())
            .build()?,
        ))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        assert_eq_or_internal_err!(
            self.left.output_partitioning().partition_count(),
            1,
            "Invalid NestedLoopJoinExec, the output partition count of the left child must be 1,\
                 consider using CoalescePartitionsExec or the EnforceDistribution rule"
        );

        let metrics = NestedLoopJoinMetrics::new(&self.metrics, partition);
        let batch_size = context.session_config().batch_size();

        // update column indices to reflect the projection
        let column_indices_after_projection = match self.projection.as_ref() {
            Some(projection) => projection
                .iter()
                .map(|i| self.column_indices[*i].clone())
                .collect(),
            None => self.column_indices.clone(),
        };

        let right_partition_count = self.right().output_partitioning().partition_count();

        // Always try to buffer all left data in memory via OnceFut.
        // If that fails with OOM, the stream will fallback to memory-limited
        // mode (if conditions allow).
        let load_reservation =
            MemoryConsumer::new(format!("NestedLoopJoinLoad[{partition}]"))
                .register(context.memory_pool());

        let build_side_data = self.build_side_data.try_once(|| {
            let stream = self.left.execute(0, Arc::clone(&context))?;

            Ok(collect_left_input(
                stream,
                metrics.join_metrics.clone(),
                load_reservation,
                need_produce_result_in_final(self.join_type),
                right_partition_count,
            ))
        })?;

        let probe_side_data = self.right.execute(partition, Arc::clone(&context))?;

        // Determine if OOM fallback to memory-limited mode is possible.
        // Conditions:
        // 1. Disk manager supports temp files (needed for spilling).
        // 2. FULL join with multiple right partitions is not yet supported
        //    in the fallback path. FULL join needs to track BOTH left-side
        //    matches (for unmatched left rows) AND right-side matches (for
        //    unmatched right rows). The fallback path builds a per-partition
        //    `JoinLeftData` with `probe_threads_counter == 1`, so each
        //    partition emits unmatched left rows based only on its own
        //    right-side matches, producing incorrect duplicate output for
        //    left rows that match in another partition. Other join types
        //    that need only one-sided final emission (LEFT, LEFT SEMI,
        //    LEFT ANTI, LEFT MARK) have a similar latent issue in the
        //    fallback path which predates this change; tracking is out of
        //    scope for this PR.
        let full_join_multi_partition =
            matches!(self.join_type, JoinType::Full) && right_partition_count > 1;
        let spill_state = if context.runtime_env().disk_manager.tmp_files_enabled()
            && !full_join_multi_partition
        {
            SpillState::Pending {
                left_plan: Arc::clone(&self.left),
                task_context: Arc::clone(&context),
                left_spill_data: Arc::clone(&self.left_spill_data),
            }
        } else {
            SpillState::Disabled
        };

        Ok(Box::pin(NestedLoopJoinStream::new(
            self.schema(),
            self.filter.clone(),
            self.join_type,
            probe_side_data,
            build_side_data,
            column_indices_after_projection,
            metrics,
            batch_size,
            spill_state,
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, partition: Option<usize>) -> Result<Arc<Statistics>> {
        // NestedLoopJoinExec is designed for joins without equijoin keys in the
        // ON clause (e.g., `t1 JOIN t2 ON (t1.v1 + t2.v1) % 2 = 0`). Any join
        // predicates are stored in `self.filter`, but `estimate_join_statistics`
        // currently doesn't support selectivity estimation for such arbitrary
        // filter expressions. We pass an empty join column list, which means
        // the cardinality estimation cannot use column statistics and returns
        // unknown row counts.
        let join_columns = Vec::new();

        // Left side is always a single partition (Distribution::SinglePartition),
        // so we always request overall stats with `None`. Right side can have
        // multiple partitions, so we forward the partition parameter to get
        // partition-specific statistics when requested.
        let left_stats = Arc::unwrap_or_clone(self.left.partition_statistics(None)?);
        let right_stats = Arc::unwrap_or_clone(match partition {
            Some(partition) => self.right.partition_statistics(Some(partition))?,
            None => self.right.partition_statistics(None)?,
        });

        let stats = estimate_join_statistics(
            left_stats,
            right_stats,
            &join_columns,
            &self.join_type,
            &self.join_schema,
        )?;

        Ok(Arc::new(stats.project(self.projection.as_ref())))
    }

    /// Tries to push `projection` down through `nested_loop_join`. If possible, performs the
    /// pushdown and returns a new [`NestedLoopJoinExec`] as the top plan which has projections
    /// as its children. Otherwise, returns `None`.
    fn try_swapping_with_projection(
        &self,
        projection: &ProjectionExec,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        // TODO: currently if there is projection in NestedLoopJoinExec, we can't push down projection to left or right input. Maybe we can pushdown the mixed projection later.
        if self.contains_projection() {
            return Ok(None);
        }

        let schema = self.schema();
        if let Some(JoinData {
            projected_left_child,
            projected_right_child,
            join_filter,
            ..
        }) = try_pushdown_through_join(
            projection,
            self.left(),
            self.right(),
            &[],
            &schema,
            self.filter(),
        )? {
            Ok(Some(Arc::new(NestedLoopJoinExec::try_new(
                Arc::new(projected_left_child),
                Arc::new(projected_right_child),
                join_filter,
                self.join_type(),
                // Returned early if projection is not None
                None,
            )?)))
        } else {
            try_embed_projection(projection, self)
        }
    }
}

impl EmbeddedProjection for NestedLoopJoinExec {
    fn with_projection(&self, projection: Option<Vec<usize>>) -> Result<Self> {
        self.with_projection(projection)
    }
}

/// Left (build-side) data
pub(crate) struct JoinLeftData {
    /// Build-side data kept as the original (un-concatenated) batches.
    ///
    /// Rows are addressed by a flat index into the logical concatenation of these
    /// batches; [`JoinLeftData::locate`] resolves a flat index to a `(batch, row)`
    /// pair. Keeping the batches separate avoids copying the whole build side into
    /// one contiguous batch.
    batches: Vec<RecordBatch>,
    /// Build-side schema. Retained so it is available even when `batches` is empty.
    schema: SchemaRef,
    /// Prefix-sum offsets: `offsets[k]` is the first flat row index of `batches[k]`,
    /// and `offsets[batches.len()]` is the total row count.
    offsets: Vec<usize>,
    /// Shared bitmap builder for visited left indices (flat-indexed)
    bitmap: SharedBitmapBuilder,
    /// Counter of running probe-threads, potentially able to update `bitmap`
    probe_threads_counter: AtomicUsize,
    /// Memory reservation for tracking batch and bitmap
    /// Cleared on `JoinLeftData` drop
    /// reservation is cleared on Drop
    #[expect(dead_code)]
    reservation: MemoryReservation,
}

impl JoinLeftData {
    pub(crate) fn new(
        batches: Vec<RecordBatch>,
        schema: SchemaRef,
        offsets: Vec<usize>,
        bitmap: SharedBitmapBuilder,
        probe_threads_counter: AtomicUsize,
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            batches,
            schema,
            offsets,
            bitmap,
            probe_threads_counter,
            reservation,
        }
    }

    /// Total number of build-side rows across all batches
    pub(crate) fn num_rows(&self) -> usize {
        *self.offsets.last().unwrap()
    }

    /// Build-side schema
    pub(crate) fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The build-side batch at index `batch_idx`
    pub(crate) fn batch_at(&self, batch_idx: usize) -> &RecordBatch {
        &self.batches[batch_idx]
    }

    /// Flat-index offsets (see field docs)
    pub(crate) fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Resolve a flat build-side row index to a `(batch, row)` pair
    pub(crate) fn locate(&self, flat: usize) -> (usize, usize) {
        flat_index_to_batch_row(&self.offsets, flat)
    }

    /// Gather the build-side rows at the given flat indices into a single batch.
    /// Single batch: a zero-copy `take`; multiple batches: gathered with
    /// `interleave` (the build side is not concatenated).
    pub(crate) fn gather(&self, indices: &UInt32Array) -> Result<RecordBatch> {
        // A zero-column build side (e.g. a decorrelated subquery that only
        // contributes row multiplicity) cannot go through `take`/`interleave`:
        // with no columns the row count cannot be inferred, so build the
        // zero-column batch with an explicit row count instead.
        if self.schema.fields().is_empty() {
            return create_record_batch_with_empty_schema(
                Arc::clone(&self.schema),
                indices.len(),
            );
        }
        if self.batches.len() == 1 {
            Ok(take_record_batch(&self.batches[0], indices)?)
        } else {
            let pairs: Vec<(usize, usize)> = indices
                .values()
                .iter()
                .map(|&i| flat_index_to_batch_row(&self.offsets, i as usize))
                .collect();
            let refs: Vec<&RecordBatch> = self.batches.iter().collect();
            Ok(interleave_record_batch(&refs, &pairs)?)
        }
    }

    pub(crate) fn bitmap(&self) -> &SharedBitmapBuilder {
        &self.bitmap
    }

    /// Decrements counter of running threads, and returns `true`
    /// if caller is the last running thread
    pub(crate) fn report_probe_completed(&self) -> bool {
        self.probe_threads_counter.fetch_sub(1, Ordering::Relaxed) == 1
    }
}

/// Asynchronously collect input into a single batch, and creates `JoinLeftData` from it
async fn collect_left_input(
    stream: SendableRecordBatchStream,
    join_metrics: BuildProbeJoinMetrics,
    reservation: MemoryReservation,
    with_visited_left_side: bool,
    probe_threads_count: usize,
) -> Result<JoinLeftData> {
    let schema = stream.schema();

    // Load all batches and count the rows
    let (batches, metrics, reservation) = stream
        .try_fold(
            (Vec::new(), join_metrics, reservation),
            |(mut batches, metrics, reservation), batch| async {
                let batch_size = batch.get_array_memory_size();
                // Reserve memory for incoming batch
                reservation.try_grow(batch_size)?;
                // Update metrics
                metrics.build_mem_used.add(batch_size);
                metrics.build_input_batches.add(1);
                metrics.build_input_rows.add(batch.num_rows());
                // Push batch to output
                batches.push(batch);
                Ok((batches, metrics, reservation))
            },
        )
        .await?;

    // Keep the build-side batches un-concatenated; rows are addressed by a flat
    // index into their logical concatenation via these prefix-sum offsets.
    let offsets = build_batch_offsets(batches.iter().map(RecordBatch::num_rows));
    let num_rows = *offsets.last().unwrap();

    // Reserve memory for visited_left_side bitmap if required by join type
    let visited_left_side = if with_visited_left_side {
        let n_rows = num_rows;
        let buffer_size = n_rows.div_ceil(8);
        reservation.try_grow(buffer_size)?;
        metrics.build_mem_used.add(buffer_size);

        let mut buffer = BooleanBufferBuilder::new(n_rows);
        buffer.append_n(n_rows, false);
        buffer
    } else {
        BooleanBufferBuilder::new(0)
    };

    Ok(JoinLeftData::new(
        batches,
        schema,
        offsets,
        Mutex::new(visited_left_side),
        AtomicUsize::new(probe_threads_count),
        reservation,
    ))
}

/// States for join processing. See `poll_next()` comment for more details about
/// state transitions.
#[derive(Debug, Clone, Copy)]
enum NLJState {
    BufferingLeft,
    FetchingRight,
    ProbeRight,
    EmitRightUnmatched,
    EmitLeftUnmatched,
    /// Emit unmatched right rows using the global bitmap accumulated across
    /// all left chunks. Only used in memory-limited mode for join types that
    /// require tracking right-side matches in the final output (RIGHT, FULL,
    /// RIGHT SEMI, RIGHT ANTI, RIGHT MARK).
    EmitGlobalRightUnmatched,
    Done,
}
/// Shared data for the left-side spill fallback.
///
/// When the in-memory `OnceFut` path fails with OOM, the first partition
/// spills the entire left side to disk. This struct holds the spill file
/// reference so other partitions can read from the same file.
pub(crate) struct LeftSpillData {
    /// SpillManager used to read the spill file (has the left schema)
    spill_manager: SpillManager,
    /// The spill file containing all left-side batches
    spill_file: RefCountedTempFile,
    /// Left-side schema
    schema: SchemaRef,
}

/// Tracks the state of the memory-limited spill fallback for NLJ.
///
/// The NLJ always starts with the standard OnceFut path. If the in-memory
/// load fails with OOM and conditions allow, the operator falls back to a
/// multi-pass strategy where left data is loaded in chunks and the right
/// side is spilled to disk.
pub(crate) enum SpillState {
    /// Fallback is not possible (e.g., join type requires global right bitmap,
    /// or disk manager is disabled). OOM errors will propagate as-is.
    Disabled,

    /// Fallback is possible but not yet triggered. The operator is still
    /// attempting the standard OnceFut path. Holds the context needed to
    /// initiate fallback if OOM occurs.
    Pending {
        /// Left child plan for re-execution
        left_plan: Arc<dyn ExecutionPlan>,
        /// TaskContext for re-execution and SpillManager creation
        task_context: Arc<TaskContext>,
        /// Shared OnceAsync for left-side spill data. The first partition
        /// to initiate fallback spills the left side; others share the file.
        left_spill_data: Arc<OnceAsync<LeftSpillData>>,
    },

    /// Fallback has been triggered. Left data is being loaded in chunks
    /// and the right side is spilled to disk for re-scanning.
    Active(Box<SpillStateActive>),
}

/// State for active memory-limited spill execution.
/// Boxed inside [`SpillState::Active`] to reduce enum size.
pub(crate) struct SpillStateActive {
    /// Shared future for left-side spill data. All partitions wait on
    /// the same future — the first to poll triggers the actual spill.
    left_spill_fut: OnceFut<LeftSpillData>,
    /// Left input stream for incremental chunk reading (from spill file).
    /// None until `left_spill_fut` resolves.
    left_stream: Option<SendableRecordBatchStream>,
    /// Left-side schema (set once `left_spill_fut` resolves)
    left_schema: Option<SchemaRef>,
    /// Memory reservation for left-side buffering
    reservation: MemoryReservation,
    /// Accumulated left batches for the current chunk
    pending_batches: Vec<RecordBatch>,
    /// Right input that spills on the first pass and replays from spill later.
    right_input: ReplayableStreamSource,
    /// Per-batch accumulated right bitmaps across all left chunks.
    /// Index = right batch sequence number (0-based, non-empty batches only).
    /// Only populated when `should_track_unmatched_right` is true.
    global_right_bitmaps: Vec<BooleanBuffer>,
    /// Separate reservation for `global_right_bitmaps`. These buffers live
    /// for the full operator lifetime (not per-chunk), so they must be
    /// tracked separately from `reservation`, which gets `resize(0)`-ed
    /// between chunks.
    global_right_bitmaps_reservation: MemoryReservation,
    /// Current right batch sequence index within the current pass.
    right_batch_index: usize,
}

impl SpillStateActive {
    /// Merge a per-pass right bitmap into the global accumulator at the
    /// given batch index, growing the dedicated reservation when seeing
    /// a batch index for the first time.
    ///
    /// On first encounter of `idx`, the bitmap is stored as-is and its
    /// size is reserved. On subsequent encounters (later left chunk
    /// passes over the same right batch), the existing entry is OR-merged
    /// with `values`. Because `bitor` produces a buffer of the same bit
    /// length, the reservation does not need to be adjusted on merge.
    fn merge_current_right_bitmap(&mut self, idx: usize, values: BooleanBuffer) {
        if idx >= self.global_right_bitmaps.len() {
            // First encounter of this right batch — account memory and store.
            // The bitmap has one bit per right row, so for very large right
            // inputs the accumulated size can be non-negligible (e.g.,
            // 1M rows ≈ 125 KB per batch).
            // Use infallible `grow` because we must accept the bitmap to
            // preserve correctness — the fallback path has no other recourse.
            let bytes = values.len().div_ceil(8);
            self.global_right_bitmaps_reservation.grow(bytes);
            self.global_right_bitmaps.push(values);
        } else {
            // Subsequent left chunk pass — OR merge. Same bit length, so
            // no reservation adjustment is needed.
            self.global_right_bitmaps[idx] =
                self.global_right_bitmaps[idx].bitor(&values);
        }
    }
}

pub(crate) struct NestedLoopJoinStream {
    // ========================================================================
    // PROPERTIES:
    // Operator's properties that remain constant
    //
    // Note: The implementation uses the terms left/build-side table and
    // right/probe-side table interchangeably. Treating the left side as the
    // build side is a convention in DataFusion: the planner always tries to
    // swap the smaller table to the left side.
    // ========================================================================
    /// Output schema
    pub(crate) output_schema: Arc<Schema>,
    /// join filter
    pub(crate) join_filter: Option<JoinFilter>,
    /// type of the join
    pub(crate) join_type: JoinType,
    /// the probe-side(right) table data of the nested loop join
    /// `Option` is used because memory-limited path requires resetting it.
    pub(crate) right_data: Option<SendableRecordBatchStream>,
    /// the build-side table data of the nested loop join
    pub(crate) left_data: OnceFut<JoinLeftData>,
    /// Projection to construct the output schema from the left and right tables.
    /// Example:
    /// - output_schema: ['a', 'c']
    /// - left_schema: ['a', 'b']
    /// - right_schema: ['c']
    ///
    /// The column indices would be [(left, 0), (right, 0)] -- taking the left
    /// 0th column and right 0th column can construct the output schema.
    ///
    /// Note there are other columns ('b' in the example) still kept after
    /// projection pushdown; this is because they might be used to evaluate
    /// the join filter (e.g., `JOIN ON (b+c)>0`).
    pub(crate) column_indices: Vec<ColumnIndex>,
    /// Join execution metrics
    pub(crate) metrics: NestedLoopJoinMetrics,

    /// `batch_size` from configuration
    batch_size: usize,

    /// See comments in [`need_produce_right_in_final`] for more detail
    should_track_unmatched_right: bool,

    // ========================================================================
    // STATE FLAGS/BUFFERS:
    // Fields that hold intermediate data/flags during execution
    // ========================================================================
    /// State Tracking
    state: NLJState,
    /// Output buffer holds the join result to output. It will emit eagerly when
    /// the threshold is reached.
    output_buffer: Box<BatchCoalescer>,
    /// Indices of output columns whose data type stores variable-length data
    /// behind `i32` offsets (e.g. `Utf8`, `Binary`, `List`), possibly nested.
    /// Used by [`Self::push_output_batch`] to keep `output_buffer` from
    /// accumulating more than `i32::MAX` bytes per such column.
    var_len_output_columns: Vec<usize>,
    /// For each column in `var_len_output_columns`, an upper-bound estimate of
    /// the variable-length bytes buffered in `output_buffer` since its buffer
    /// was last finished. The estimate never resets on the coalescer's internal
    /// batch completions, so it can only overcount (forcing a slightly early
    /// finish), never undercount.
    buffered_var_len_bytes: Vec<usize>,
    /// See comments in [`NLJState::Done`] for its purpose
    handled_empty_output: bool,

    // Buffer(left) side
    // -----------------
    /// The current buffered left data to join
    buffered_left_data: Option<Arc<JoinLeftData>>,
    /// Index into the left buffered batch. Used in `ProbeRight` state
    left_probe_idx: usize,
    /// Index into the left buffered batch. Used in `EmitLeftUnmatched` state
    left_emit_idx: usize,
    /// Should we go back to `BufferingLeft` state again after `EmitLeftUnmatched`
    /// state is over.
    left_exhausted: bool,
    /// If we can buffer all left data in one pass (false means memory-limited multi-pass)
    left_buffered_in_one_pass: bool,

    // Probe(right) side
    // -----------------
    /// The current probe batch to process
    current_right_batch: Option<RecordBatch>,
    // For right join, keep track of matched rows in `current_right_batch`
    // Constructed when fetching each new incoming right batch in `FetchingRight` state.
    current_right_batch_matched: Option<BooleanArray>,

    /// Memory-limited spill fallback state. See [`SpillState`] for details.
    spill_state: SpillState,
}

pub(crate) struct NestedLoopJoinMetrics {
    /// Join execution metrics
    pub(crate) join_metrics: BuildProbeJoinMetrics,
    /// Selectivity of the join: output_rows / (left_rows * right_rows)
    pub(crate) selectivity: RatioMetrics,
    /// Spill metrics for memory-limited execution
    pub(crate) spill_metrics: SpillMetrics,
}

impl NestedLoopJoinMetrics {
    pub fn new(metrics: &ExecutionPlanMetricsSet, partition: usize) -> Self {
        Self {
            join_metrics: BuildProbeJoinMetrics::new(partition, metrics),
            selectivity: MetricBuilder::new(metrics)
                .with_type(MetricType::Summary)
                .ratio_metrics("selectivity", partition),
            spill_metrics: SpillMetrics::new(metrics, partition),
        }
    }
}

impl Stream for NestedLoopJoinStream {
    type Item = Result<RecordBatch>;

    /// See the comments [`NestedLoopJoinExec`] for high-level design ideas.
    ///
    /// # Implementation
    ///
    /// This function is the entry point of NLJ operator's state machine
    /// transitions. The rough state transition graph is as follow, for more
    /// details see the comment in each state's matching arm.
    ///
    /// ============================
    /// State transition graph:
    /// ============================
    ///
    /// (start) --> BufferingLeft
    /// ----------------------------
    /// BufferingLeft → FetchingRight
    ///
    /// FetchingRight → ProbeRight (if right batch available)
    /// FetchingRight → EmitLeftUnmatched (if right exhausted)
    ///
    /// ProbeRight → ProbeRight (next left row or after yielding output)
    /// ProbeRight → EmitRightUnmatched (for special join types like right join)
    /// ProbeRight → FetchingRight (done with the current right batch)
    ///
    /// EmitRightUnmatched → FetchingRight
    ///
    /// EmitLeftUnmatched → EmitLeftUnmatched (only process 1 chunk for each
    /// iteration)
    /// EmitLeftUnmatched → Done (if finished)
    /// ----------------------------
    /// Done → (end)
    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        loop {
            match self.state {
                // # NLJState transitions
                // --> FetchingRight
                // This state will prepare the left side batches, next state
                // `FetchingRight` is responsible for preparing a single probe
                // side batch, before start joining.
                NLJState::BufferingLeft => {
                    debug!("[NLJState] Entering: {:?}", self.state);
                    // inside `collect_left_input` (the routine to buffer build
                    // -side batches), related metrics except build time will be
                    // updated.
                    // stop on drop
                    let build_metric = self.metrics.join_metrics.build_time.clone();
                    let _build_timer = build_metric.timer();

                    match self.handle_buffering_left(cx) {
                        ControlFlow::Continue(()) => continue,
                        ControlFlow::Break(poll) => return poll,
                    }
                }

                // # NLJState transitions:
                // 1. --> ProbeRight
                //    Start processing the join for the newly fetched right
                //    batch.
                // 2. --> EmitLeftUnmatched: When the right side input is exhausted, (maybe) emit
                //    unmatched left side rows.
                //
                // After fetching a new batch from the right side, it will
                // process all rows from the buffered left data:
                // ```text
                // for batch in right_side:
                //     for row in left_buffer:
                //         join(batch, row)
                // ```
                // Note: the implementation does this step incrementally,
                // instead of materializing all intermediate Cartesian products
                // at once in memory.
                //
                // So after the right side input is exhausted, the join phase
                // for the current buffered left data is finished. We can go to
                // the next `EmitLeftUnmatched` phase to check if there is any
                // special handling (e.g., in cases like left join).
                NLJState::FetchingRight => {
                    debug!("[NLJState] Entering: {:?}", self.state);
                    // stop on drop
                    let join_metric = self.metrics.join_metrics.join_time.clone();
                    let _join_timer = join_metric.timer();

                    match self.handle_fetching_right(cx) {
                        ControlFlow::Continue(()) => continue,
                        ControlFlow::Break(poll) => return poll,
                    }
                }

                // NLJState transitions:
                // 1. --> ProbeRight(1)
                //    If we have already buffered enough output to yield, it
                //    will first give back control to the parent state machine,
                //    then resume at the same place.
                // 2. --> ProbeRight(2)
                //    After probing one right batch, and evaluating the
                //    join filter on (left-row x right-batch), it will advance
                //    to the next left row, then re-enter the current state and
                //    continue joining.
                // 3. --> FetchRight
                //    After it has done with the current right batch (to join
                //    with all rows in the left buffer), it will go to
                //    FetchRight state to check what to do next.
                NLJState::ProbeRight => {
                    debug!("[NLJState] Entering: {:?}", self.state);

                    // stop on drop
                    let join_metric = self.metrics.join_metrics.join_time.clone();
                    let _join_timer = join_metric.timer();

                    match self.handle_probe_right() {
                        ControlFlow::Continue(()) => continue,
                        ControlFlow::Break(poll) => {
                            return self.metrics.join_metrics.baseline.record_poll(poll);
                        }
                    }
                }

                // In the `current_right_batch_matched` bitmap, all trues mean
                // it has been output by the join. In this state we have to
                // output unmatched rows for current right batch (with null
                // padding for left relation)
                // Precondition: we have checked the join type so that it's
                // possible to output right unmatched (e.g. it's right join)
                NLJState::EmitRightUnmatched => {
                    debug!("[NLJState] Entering: {:?}", self.state);

                    // stop on drop
                    let join_metric = self.metrics.join_metrics.join_time.clone();
                    let _join_timer = join_metric.timer();

                    match self.handle_emit_right_unmatched() {
                        ControlFlow::Continue(()) => continue,
                        ControlFlow::Break(poll) => {
                            return self.metrics.join_metrics.baseline.record_poll(poll);
                        }
                    }
                }

                // NLJState transitions:
                // 1. --> EmitLeftUnmatched(1)
                //    If we have already buffered enough output to yield, it
                //    will first give back control to the parent state machine,
                //    then resume at the same place.
                // 2. --> EmitLeftUnmatched(2)
                //    After processing some unmatched rows, it will re-enter
                //    the same state, to check if there are any more final
                //    results to output.
                // 3. --> Done
                //    It has processed all data, go to the final state and ready
                //    to exit.
                // 4. --> BufferingLeft (memory-limited mode only)
                //    When left data was loaded in chunks and more chunks remain,
                //    go back to BufferingLeft to load the next chunk.
                NLJState::EmitLeftUnmatched => {
                    debug!("[NLJState] Entering: {:?}", self.state);

                    // stop on drop
                    let join_metric = self.metrics.join_metrics.join_time.clone();
                    let _join_timer = join_metric.timer();

                    match self.handle_emit_left_unmatched() {
                        ControlFlow::Continue(()) => continue,
                        ControlFlow::Break(poll) => {
                            return self.metrics.join_metrics.baseline.record_poll(poll);
                        }
                    }
                }

                // Replay all right batches from spill and emit unmatched
                // right rows using the global bitmap accumulated across all
                // left chunks. Only entered in memory-limited mode for join
                // types where `should_track_unmatched_right` is true
                // (RIGHT, FULL, RIGHT SEMI, RIGHT ANTI, RIGHT MARK).
                NLJState::EmitGlobalRightUnmatched => {
                    debug!("[NLJState] Entering: {:?}", self.state);

                    let join_metric = self.metrics.join_metrics.join_time.clone();
                    let _join_timer = join_metric.timer();

                    match self.handle_emit_global_right_unmatched(cx) {
                        ControlFlow::Continue(()) => continue,
                        ControlFlow::Break(poll) => {
                            return self.metrics.join_metrics.baseline.record_poll(poll);
                        }
                    }
                }

                // The final state and the exit point
                NLJState::Done => {
                    debug!("[NLJState] Entering: {:?}", self.state);

                    // stop on drop
                    let join_metric = self.metrics.join_metrics.join_time.clone();
                    let _join_timer = join_metric.timer();
                    // counting it in join timer due to there might be some
                    // final resout batches to output in this state

                    let poll = self.handle_done();
                    return self.metrics.join_metrics.baseline.record_poll(poll);
                }
            }
        }
    }
}

impl RecordBatchStream for NestedLoopJoinStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }
}

impl NestedLoopJoinStream {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        schema: Arc<Schema>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        right_data: SendableRecordBatchStream,
        left_data: OnceFut<JoinLeftData>,
        column_indices: Vec<ColumnIndex>,
        metrics: NestedLoopJoinMetrics,
        batch_size: usize,
        spill_state: SpillState,
    ) -> Self {
        let var_len_output_columns: Vec<usize> = schema
            .fields()
            .iter()
            .enumerate()
            .filter(|(_, field)| contains_i32_offset_data(field.data_type()))
            .map(|(idx, _)| idx)
            .collect();
        let buffered_var_len_bytes = vec![0; var_len_output_columns.len()];
        Self {
            output_schema: Arc::clone(&schema),
            join_filter: filter,
            join_type,
            right_data: Some(right_data),
            column_indices,
            left_data,
            metrics,
            buffered_left_data: None,
            output_buffer: Box::new(BatchCoalescer::new(schema, batch_size)),
            var_len_output_columns,
            buffered_var_len_bytes,
            batch_size,
            current_right_batch: None,
            current_right_batch_matched: None,
            state: NLJState::BufferingLeft,
            left_probe_idx: 0,
            left_emit_idx: 0,
            left_exhausted: false,
            left_buffered_in_one_pass: true,
            handled_empty_output: false,
            should_track_unmatched_right: need_produce_right_in_final(join_type),
            spill_state,
        }
    }

    /// Returns true if this stream is operating in memory-limited mode
    fn is_memory_limited(&self) -> bool {
        matches!(self.spill_state, SpillState::Active(_))
    }

    /// Check if we can fall back to memory-limited mode on this error.
    fn can_fallback_to_spill(&self, error: &datafusion_common::DataFusionError) -> bool {
        matches!(self.spill_state, SpillState::Pending { .. })
            && matches!(
                error.find_root(),
                datafusion_common::DataFusionError::ResourcesExhausted(_)
            )
    }

    /// Switch from the standard OnceFut path to memory-limited mode.
    ///
    /// Uses the shared `left_spill_data` OnceAsync so that only the first
    /// partition to reach this point re-executes the left child and spills
    /// it to disk. Other partitions share the same spill file.
    fn initiate_fallback(&mut self) -> Result<()> {
        // Take ownership of Pending state
        let (left_plan, context, left_spill_data) =
            match std::mem::replace(&mut self.spill_state, SpillState::Disabled) {
                SpillState::Pending {
                    left_plan,
                    task_context,
                    left_spill_data,
                } => (left_plan, task_context, left_spill_data),
                _ => {
                    return internal_err!(
                        "initiate_fallback called in non-Pending spill state"
                    );
                }
            };

        // Use OnceAsync to ensure only the first partition spills the left
        // side. Other partitions will get the same OnceFut that resolves
        // to the shared spill file.
        let left_spill_fut = left_spill_data.try_once(|| {
            let plan = Arc::clone(&left_plan);
            let ctx = Arc::clone(&context);
            let spill_metrics = self.metrics.spill_metrics.clone();
            Ok(async move {
                let mut stream = plan.execute(0, Arc::clone(&ctx))?;
                let schema = stream.schema();
                let left_spill_manager = SpillManager::new(
                    ctx.runtime_env(),
                    spill_metrics,
                    Arc::clone(&schema),
                )
                .with_compression_type(ctx.session_config().spill_compression());

                let result = left_spill_manager
                    .spill_record_batch_stream_and_return_max_batch_memory(
                        &mut stream,
                        "NestedLoopJoin left spill",
                    )
                    .await?;

                match result {
                    Some((file, _max_batch_memory)) => Ok(LeftSpillData {
                        spill_manager: left_spill_manager,
                        spill_file: file,
                        schema,
                    }),
                    None => {
                        internal_err!("Left side produced no data to spill")
                    }
                }
            })
        })?;

        // Create reservation with can_spill for fair memory allocation
        let reservation = MemoryConsumer::new("NestedLoopJoinLoad[fallback]".to_string())
            .with_can_spill(true)
            .register(context.memory_pool());

        // Separate reservation for the global right bitmaps. These buffers
        // persist across all left chunks, whereas `reservation` is reset
        // between chunks via `resize(0)`.
        let global_right_bitmaps_reservation =
            MemoryConsumer::new("NestedLoopJoinGlobalRightBitmaps".to_string())
                .register(context.memory_pool());

        // Create SpillManager for right-side spilling
        let right_schema = self
            .right_data
            .as_ref()
            .expect("right_data must be present before fallback")
            .schema();
        let right_data = self
            .right_data
            .take()
            .expect("right_data must be present before fallback");
        let right_spill_manager = SpillManager::new(
            context.runtime_env(),
            self.metrics.spill_metrics.clone(),
            right_schema,
        )
        .with_compression_type(context.session_config().spill_compression());

        self.spill_state = SpillState::Active(Box::new(SpillStateActive {
            left_spill_fut,
            left_stream: None,
            left_schema: None,
            reservation,
            pending_batches: Vec::new(),
            right_input: ReplayableStreamSource::new(
                right_data,
                right_spill_manager,
                "NestedLoopJoin right spill",
            ),
            global_right_bitmaps: Vec::new(),
            global_right_bitmaps_reservation,
            right_batch_index: 0,
        }));

        // State stays BufferingLeft — next poll will enter
        // handle_buffering_left_memory_limited via is_memory_limited() check
        self.state = NLJState::BufferingLeft;

        Ok(())
    }

    // ==== State handler functions ====

    /// Handle BufferingLeft state - prepare left side batches.
    ///
    /// In standard mode, uses OnceFut to load all left data at once.
    /// In memory-limited mode, incrementally buffers left batches until the
    /// memory budget is reached or the left stream is exhausted.
    fn handle_buffering_left(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> ControlFlow<Poll<Option<Result<RecordBatch>>>> {
        if self.is_memory_limited() {
            self.handle_buffering_left_memory_limited(cx)
        } else {
            // Standard path: use OnceFut
            match self.left_data.get_shared(cx) {
                Poll::Ready(Ok(left_data)) => {
                    self.buffered_left_data = Some(left_data);
                    self.left_exhausted = true;
                    self.state = NLJState::FetchingRight;
                    ControlFlow::Continue(())
                }
                Poll::Ready(Err(e)) => {
                    if self.can_fallback_to_spill(&e) {
                        debug!(
                            "NestedLoopJoin: OnceFut failed with OOM, \
                             falling back to memory-limited mode"
                        );
                        match self.initiate_fallback() {
                            Ok(()) => ControlFlow::Continue(()),
                            Err(fallback_err) => {
                                ControlFlow::Break(Poll::Ready(Some(Err(fallback_err))))
                            }
                        }
                    } else {
                        ControlFlow::Break(Poll::Ready(Some(Err(e))))
                    }
                }
                Poll::Pending => ControlFlow::Break(Poll::Pending),
            }
        }
    }

    /// Memory-limited path for handle_buffering_left.
    ///
    /// Incrementally polls the left stream and accumulates batches until:
    /// - Memory reservation fails (chunk is full, more data remains)
    /// - Left stream is exhausted (this is the last/only chunk)
    fn handle_buffering_left_memory_limited(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> ControlFlow<Poll<Option<Result<RecordBatch>>>> {
        let SpillState::Active(active) = &mut self.spill_state else {
            unreachable!(
                "handle_buffering_left_memory_limited called without Active spill state"
            );
        };

        // On first entry (or after re-entry for a new chunk pass when
        // left_stream was consumed), wait for the shared left spill
        // future to resolve and then open a stream from the spill file.
        if active.left_stream.is_none() {
            match active.left_spill_fut.get_shared(cx) {
                Poll::Ready(Ok(spill_data)) => {
                    match spill_data
                        .spill_manager
                        .read_spill_as_stream(spill_data.spill_file.clone(), None)
                    {
                        Ok(stream) => {
                            active.left_schema = Some(Arc::clone(&spill_data.schema));
                            active.left_stream = Some(stream);
                        }
                        Err(e) => {
                            return ControlFlow::Break(Poll::Ready(Some(Err(e))));
                        }
                    }
                }
                Poll::Ready(Err(e)) => {
                    return ControlFlow::Break(Poll::Ready(Some(Err(e))));
                }
                Poll::Pending => {
                    return ControlFlow::Break(Poll::Pending);
                }
            }
        }

        let left_stream = active
            .left_stream
            .as_mut()
            .expect("left_stream must be set after spill future resolves");

        // Poll left stream for more batches.
        // Note: pending_batches may already contain a batch from the
        // previous chunk iteration (the batch that triggered the memory limit).
        loop {
            match left_stream.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(batch))) => {
                    if batch.num_rows() == 0 {
                        continue;
                    }
                    let batch_rows = batch.num_rows();
                    let batch_size = batch.get_array_memory_size();
                    let can_grow = active.reservation.try_grow(batch_size).is_ok();

                    if !can_grow && !active.pending_batches.is_empty() {
                        // Memory limit reached and we already have data.
                        // Push this batch into pending (it's already in memory)
                        // and stop buffering for this chunk.
                        active.pending_batches.push(batch);
                        self.left_exhausted = false;
                        self.left_buffered_in_one_pass = false;
                        break;
                    } else if !can_grow {
                        // No pending batches yet — we must accept this batch
                        // to make progress, even if it exceeds the budget.
                        active.reservation.grow(batch_size);
                    }

                    self.metrics.join_metrics.build_mem_used.add(batch_size);
                    self.metrics.join_metrics.build_input_batches.add(1);
                    self.metrics.join_metrics.build_input_rows.add(batch_rows);
                    active.pending_batches.push(batch);
                }
                Poll::Ready(Some(Err(e))) => {
                    return ControlFlow::Break(Poll::Ready(Some(Err(e))));
                }
                Poll::Ready(None) => {
                    // Left stream exhausted
                    self.left_exhausted = true;
                    break;
                }
                Poll::Pending => {
                    return ControlFlow::Break(Poll::Pending);
                }
            }
        }

        // If the left stream is fully exhausted, release its resources so the
        // upstream pipeline can be torn down before we move on to probing.
        if self.left_exhausted {
            active.left_stream = None;
        }

        if active.pending_batches.is_empty() {
            // No data at all — go directly to Done
            self.left_exhausted = true;
            self.state = NLJState::Done;
            return ControlFlow::Continue(());
        }

        let merged_batch = match concat_batches(
            active
                .left_schema
                .as_ref()
                .expect("left_schema must be set"),
            &active.pending_batches,
        ) {
            Ok(batch) => batch,
            Err(e) => {
                return ControlFlow::Break(Poll::Ready(Some(Err(e.into()))));
            }
        };
        active.pending_batches.clear();

        // Build visited bitmap if needed for this join type
        let with_visited = need_produce_result_in_final(self.join_type);
        let n_rows = merged_batch.num_rows();
        let visited_left_side = if with_visited {
            let buffer_size = n_rows.div_ceil(8);
            // Use infallible grow for bitmap — it's small
            active.reservation.grow(buffer_size);
            self.metrics.join_metrics.build_mem_used.add(buffer_size);
            let mut buffer = BooleanBufferBuilder::new(n_rows);
            buffer.append_n(n_rows, false);
            buffer
        } else {
            BooleanBufferBuilder::new(0)
        };

        // Create an empty reservation for JoinLeftData's RAII field.
        // The actual memory tracking is managed by the Active state's reservation.
        let dummy_reservation = active.reservation.new_empty();

        // This memory-limited path already merges the pending chunk into a single
        // batch; store it as a one-element batch list so flat indices map directly.
        let left_schema = merged_batch.schema();
        let batches = vec![merged_batch];
        let offsets = build_batch_offsets(batches.iter().map(RecordBatch::num_rows));
        let left_data = JoinLeftData::new(
            batches,
            left_schema,
            offsets,
            Mutex::new(visited_left_side),
            // In memory-limited mode, only 1 probe thread per chunk
            AtomicUsize::new(1),
            dummy_reservation,
        );

        self.buffered_left_data = Some(Arc::new(left_data));

        active.right_batch_index = 0;
        match active.right_input.open_pass() {
            Ok(stream) => {
                self.right_data = Some(stream);
            }
            Err(e) => {
                return ControlFlow::Break(Poll::Ready(Some(Err(e))));
            }
        }

        self.state = NLJState::FetchingRight;
        ControlFlow::Continue(())
    }

    /// Handle FetchingRight state - fetch next right batch and prepare for processing.
    ///
    /// In memory-limited mode during the first pass, each right batch is also
    /// written to a spill file so it can be re-read on subsequent passes.
    fn handle_fetching_right(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> ControlFlow<Poll<Option<Result<RecordBatch>>>> {
        match self
            .right_data
            .as_mut()
            .expect("right_data must be present while fetching right")
            .poll_next_unpin(cx)
        {
            Poll::Ready(result) => match result {
                Some(Ok(right_batch)) => {
                    // Update metrics
                    let right_batch_rows = right_batch.num_rows();
                    self.metrics.join_metrics.input_rows.add(right_batch_rows);
                    self.metrics.join_metrics.input_batches.add(1);

                    // Skip the empty batch
                    if right_batch_rows == 0 {
                        return ControlFlow::Continue(());
                    }

                    self.current_right_batch = Some(right_batch);

                    // Prepare right bitmap
                    if self.should_track_unmatched_right {
                        let zeroed_buf = BooleanBuffer::new_unset(right_batch_rows);
                        self.current_right_batch_matched =
                            Some(BooleanArray::new(zeroed_buf, None));
                    }

                    self.left_probe_idx = 0;
                    self.state = NLJState::ProbeRight;
                    ControlFlow::Continue(())
                }
                Some(Err(e)) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
                None => {
                    self.state = NLJState::EmitLeftUnmatched;
                    ControlFlow::Continue(())
                }
            },
            Poll::Pending => ControlFlow::Break(Poll::Pending),
        }
    }

    /// Handle ProbeRight state - process current probe batch
    fn handle_probe_right(&mut self) -> ControlFlow<Poll<Option<Result<RecordBatch>>>> {
        // Return any completed batches first
        if let Some(poll) = self.maybe_flush_ready_batch() {
            return ControlFlow::Break(poll);
        }

        // Process current probe state
        match self.process_probe_batch() {
            // State unchanged (ProbeRight)
            // Continue probing until we have done joining the
            // current right batch with all buffered left rows.
            Ok(true) => ControlFlow::Continue(()),
            // To next FetchRightState
            // We have finished joining
            // (cur_right_batch x buffered_left_batches)
            Ok(false) => {
                // Left exhausted, transition to FetchingRight
                self.left_probe_idx = 0;

                // Selectivity Metric: Update total possibilities for the batch (left_rows * right_rows)
                // If memory-limited execution is implemented, this logic must be updated accordingly.
                if let (Ok(left_data), Some(right_batch)) =
                    (self.get_left_data(), self.current_right_batch.as_ref())
                {
                    let left_rows = left_data.num_rows();
                    let right_rows = right_batch.num_rows();
                    self.metrics.selectivity.add_total(left_rows * right_rows);
                }

                if self.should_track_unmatched_right {
                    debug_assert!(
                        self.current_right_batch_matched.is_some(),
                        "If it's required to track matched rows in the right input, the right bitmap must be present"
                    );
                    self.state = NLJState::EmitRightUnmatched;
                } else {
                    self.current_right_batch = None;
                    self.state = NLJState::FetchingRight;
                }
                ControlFlow::Continue(())
            }
            Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
        }
    }

    /// Handle EmitRightUnmatched state - emit unmatched right rows.
    ///
    /// In memory-limited mode, instead of emitting unmatched right rows
    /// per-batch (which would be incorrect since more left chunks may
    /// match those rows), we merge the bitmap into the global accumulator
    /// and defer emission to `EmitGlobalRightUnmatched`.
    fn handle_emit_right_unmatched(
        &mut self,
    ) -> ControlFlow<Poll<Option<Result<RecordBatch>>>> {
        // In memory-limited mode, merge bitmap into global and move on
        if self.is_memory_limited() {
            debug_assert!(
                self.current_right_batch_matched.is_some(),
                "right bitmap must be present"
            );
            let bitmap = std::mem::take(&mut self.current_right_batch_matched)
                .expect("right bitmap should be available");
            let (values, _nulls) = bitmap.into_parts();

            if let SpillState::Active(ref mut active) = self.spill_state {
                let idx = active.right_batch_index;
                active.merge_current_right_bitmap(idx, values);
                active.right_batch_index += 1;
            }

            self.current_right_batch = None;
            self.state = NLJState::FetchingRight;
            return ControlFlow::Continue(());
        }

        // Standard (single-pass) mode: emit unmatched right rows immediately
        // Return any completed batches first
        if let Some(poll) = self.maybe_flush_ready_batch() {
            return ControlFlow::Break(poll);
        }

        debug_assert!(
            self.current_right_batch_matched.is_some()
                && self.current_right_batch.is_some(),
            "This state is yielding output for unmatched rows in the current right batch, so both the right batch and the bitmap must be present"
        );
        match self.process_right_unmatched() {
            Ok(Some(batch)) => match self.push_output_batch(batch) {
                Ok(()) => {
                    debug_assert!(self.current_right_batch.is_none());
                    self.state = NLJState::FetchingRight;
                    ControlFlow::Continue(())
                }
                Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
            },
            Ok(None) => {
                debug_assert!(self.current_right_batch.is_none());
                self.state = NLJState::FetchingRight;
                ControlFlow::Continue(())
            }
            Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
        }
    }

    /// Handle EmitLeftUnmatched state - emit unmatched left rows.
    ///
    /// In memory-limited mode, after processing all unmatched rows for the
    /// current left chunk, transitions back to `BufferingLeft` to load the
    /// next chunk (if the left stream is not yet exhausted).
    fn handle_emit_left_unmatched(
        &mut self,
    ) -> ControlFlow<Poll<Option<Result<RecordBatch>>>> {
        // Return any completed batches first
        if let Some(poll) = self.maybe_flush_ready_batch() {
            return ControlFlow::Break(poll);
        }

        // Process current unmatched state
        match self.process_left_unmatched() {
            // State unchanged (EmitLeftUnmatched)
            // Continue processing until we have processed all unmatched rows
            Ok(true) => ControlFlow::Continue(()),
            // We have finished processing all unmatched rows for this chunk
            Ok(false) => match self.finish_output_buffer() {
                Ok(()) => {
                    // Flush any completed batch before transitioning.
                    // This is critical for the memory-limited path: the
                    // ProbeRight results must be emitted before we discard
                    // the current chunk and load the next one.
                    if let Some(poll) = self.maybe_flush_ready_batch() {
                        return ControlFlow::Break(poll);
                    }

                    if !self.left_exhausted && self.is_memory_limited() {
                        // More left data to process — free current chunk and
                        // go back to BufferingLeft for the next chunk
                        if let SpillState::Active(ref active) = self.spill_state {
                            active.reservation.resize(0);
                        }
                        self.buffered_left_data = None;
                        self.left_probe_idx = 0;
                        self.left_emit_idx = 0;
                        self.state = NLJState::BufferingLeft;
                    } else if self.is_memory_limited()
                        && self.should_track_unmatched_right
                    {
                        // All left chunks done — emit global right unmatched.
                        // Drop the exhausted right stream so that
                        // EmitGlobalRightUnmatched opens a fresh replay pass
                        // from the spill file. (process_left_unmatched_range
                        // already ran with right_data still set, so its
                        // schema access is not affected.)
                        self.right_data = None;
                        self.state = NLJState::EmitGlobalRightUnmatched;
                    } else {
                        self.state = NLJState::Done;
                    }
                    ControlFlow::Continue(())
                }
                Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
            },
            Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
        }
    }

    /// Handle EmitGlobalRightUnmatched state.
    ///
    /// Replays all right batches from the spill file and emits unmatched
    /// right rows using the global bitmap accumulated across all left chunks.
    fn handle_emit_global_right_unmatched(
        &mut self,
        cx: &mut std::task::Context<'_>,
    ) -> ControlFlow<Poll<Option<Result<RecordBatch>>>> {
        // Flush any completed batches first
        if let Some(poll) = self.maybe_flush_ready_batch() {
            return ControlFlow::Break(poll);
        }

        // On first entry, open a new replay pass on the right input
        if self.right_data.is_none() {
            let SpillState::Active(ref mut active) = self.spill_state else {
                unreachable!("EmitGlobalRightUnmatched without Active spill state");
            };
            active.right_batch_index = 0;
            match active.right_input.open_pass() {
                Ok(stream) => {
                    self.right_data = Some(stream);
                }
                Err(e) => {
                    return ControlFlow::Break(Poll::Ready(Some(Err(e))));
                }
            }
        }

        // Poll the replay stream for the next right batch
        match self
            .right_data
            .as_mut()
            .expect("right_data must be present")
            .poll_next_unpin(cx)
        {
            Poll::Ready(Some(Ok(right_batch))) => {
                if right_batch.num_rows() == 0 {
                    return ControlFlow::Continue(());
                }

                let SpillState::Active(ref mut active) = self.spill_state else {
                    unreachable!();
                };
                let idx = active.right_batch_index;
                active.right_batch_index += 1;

                // Build BooleanArray from the global bitmap
                let bitmap = if idx < active.global_right_bitmaps.len() {
                    BooleanArray::new(active.global_right_bitmaps[idx].clone(), None)
                } else {
                    // Batch never seen — treat all rows as unmatched
                    BooleanArray::new(
                        BooleanBuffer::new_unset(right_batch.num_rows()),
                        None,
                    )
                };

                let left_schema = Arc::clone(
                    active
                        .left_schema
                        .as_ref()
                        .expect("left_schema must be set"),
                );

                match build_unmatched_batch(
                    &self.output_schema,
                    &right_batch,
                    bitmap,
                    &left_schema,
                    &self.column_indices,
                    self.join_type,
                    JoinSide::Right,
                ) {
                    Ok(Some(batch)) => match self.push_output_batch(batch) {
                        Ok(()) => ControlFlow::Continue(()),
                        Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
                    },
                    Ok(None) => ControlFlow::Continue(()),
                    Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
                }
            }
            Poll::Ready(Some(Err(e))) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
            Poll::Ready(None) => {
                // All right batches replayed
                match self.finish_output_buffer() {
                    Ok(()) => {
                        self.state = NLJState::Done;
                        ControlFlow::Continue(())
                    }
                    Err(e) => ControlFlow::Break(Poll::Ready(Some(Err(e)))),
                }
            }
            Poll::Pending => ControlFlow::Break(Poll::Pending),
        }
    }

    /// Handle Done state - final state processing
    fn handle_done(&mut self) -> Poll<Option<Result<RecordBatch>>> {
        // Return any remaining completed batches before final termination
        if let Some(poll) = self.maybe_flush_ready_batch() {
            return poll;
        }

        // HACK for the doc test in https://github.com/apache/datafusion/blob/main/datafusion/core/src/dataframe/mod.rs#L1265
        // If this operator directly return `Poll::Ready(None)`
        // for empty result, the final result will become an empty
        // batch with empty schema, however the expected result
        // should be with the expected schema for this operator
        if !self.handled_empty_output {
            let zero_count = Count::new();
            if *self.metrics.join_metrics.baseline.output_rows() == zero_count {
                let empty_batch = RecordBatch::new_empty(Arc::clone(&self.output_schema));
                self.handled_empty_output = true;
                return Poll::Ready(Some(Ok(empty_batch)));
            }
        }

        Poll::Ready(None)
    }

    // ==== Core logic handling for each state ====

    /// Returns bool to indicate should it continue probing
    /// true -> continue in the same ProbeRight state
    /// false -> It has done with the (buffered_left x cur_right_batch), go to
    /// next state (ProbeRight)
    fn process_probe_batch(&mut self) -> Result<bool> {
        let left_data = Arc::clone(self.get_left_data()?);
        let right_batch = self
            .current_right_batch
            .as_ref()
            .ok_or_else(|| internal_datafusion_err!("Right batch should be available"))?
            .clone();

        // stop probing, the caller will go to the next state
        if self.left_probe_idx >= left_data.num_rows() {
            return Ok(false);
        }

        // ========
        // Join (l_row x right_batch)
        // and push the result into output_buffer
        // ========

        // Special case:
        // When the right batch is very small, join with multiple left rows at once,
        //
        // The regular implementation is not efficient if the plan's right child is
        // very small (e.g. 1 row total), because inside the inner loop of NLJ, it's
        // handling one input right batch at once, if it's not large enough, the
        // overheads like filter evaluation can't be amortized through vectorization.
        debug_assert_ne!(
            right_batch.num_rows(),
            0,
            "When fetching the right batch, empty batches will be skipped"
        );

        let l_row_cnt_ratio = self.batch_size / right_batch.num_rows();
        if l_row_cnt_ratio > 10 {
            // Calculate max left rows to handle at once. This operator tries to handle
            // up to `datafusion.execution.batch_size` rows at once in the intermediate
            // batch.
            let l_row_count = std::cmp::min(
                l_row_cnt_ratio,
                left_data.num_rows() - self.left_probe_idx,
            );

            debug_assert!(
                l_row_count != 0,
                "This function should only be entered when there are remaining left rows to process"
            );
            let joined_batch = self.process_left_range_join(
                &left_data,
                &right_batch,
                self.left_probe_idx,
                l_row_count,
            )?;

            if let Some(batch) = joined_batch {
                self.push_output_batch(batch)?;
            }

            self.left_probe_idx += l_row_count;

            return Ok(true);
        }

        let l_idx = self.left_probe_idx;
        let joined_batches =
            self.process_single_left_row_join(&left_data, &right_batch, l_idx)?;

        for batch in joined_batches {
            self.push_output_batch(batch)?;
        }

        // ==== Prepare for the next iteration ====

        // Advance left cursor
        self.left_probe_idx += 1;

        // Return true to continue probing
        Ok(true)
    }

    /// Process [l_start_index, l_start_index + l_count) JOIN right_batch
    /// Returns a RecordBatch containing the join results (None if empty)
    ///
    /// Side Effect: If the join type requires, left or right side matched bitmap
    /// will be set for matched indices.
    fn process_left_range_join(
        &mut self,
        left_data: &JoinLeftData,
        right_batch: &RecordBatch,
        l_start_index: usize,
        l_row_count: usize,
    ) -> Result<Option<RecordBatch>> {
        // Construct the Cartesian product between the specified range of left rows
        // and the entire right_batch. First, it calculates the index vectors, then
        // materializes the intermediate batch, and finally applies the join filter
        // to it.
        //
        // The result is capped at `batch_size` rows, so `i32` offset overflow is
        // only possible for very large variable-length values (avg >
        // `i32::MAX / batch_size` bytes per row). This path materializes via
        // `take`, which returns a proper error (not a panic) in that case.
        // -----------------------------------------------------------
        let right_rows = right_batch.num_rows();
        let total_rows = l_row_count * right_rows;

        // Build index arrays for cartesian product: left_range X right_batch
        let left_indices: UInt32Array =
            UInt32Array::from_iter_values((0..l_row_count).flat_map(|i| {
                std::iter::repeat_n((l_start_index + i) as u32, right_rows)
            }));
        let right_indices: UInt32Array = UInt32Array::from_iter_values(
            (0..l_row_count).flat_map(|_| 0..right_rows as u32),
        );

        // The build side is stored as several un-concatenated batches. Gather the
        // selected left rows into a single batch once (row `k` is the build-side row
        // at flat index `left_indices[k]`) and reuse it for both filter evaluation
        // and output construction below.
        let left_gathered = left_data.gather(&left_indices)?;

        debug_assert!(
            left_indices.len() == right_indices.len()
                && right_indices.len() == total_rows,
            "The length or cartesian product should be (left_size * right_size)",
        );

        // Evaluate the join filter (if any) over an intermediate batch built
        // using the filter's own schema/column indices.
        let bitmap_combined = if let Some(filter) = &self.join_filter {
            // Build the intermediate batch for filter evaluation
            let intermediate_batch = if filter.schema.fields().is_empty() {
                // Constant predicate (e.g., TRUE/FALSE). Use an empty schema with row_count
                create_record_batch_with_empty_schema(
                    Arc::new((*filter.schema).clone()),
                    total_rows,
                )?
            } else {
                let mut filter_columns: Vec<Arc<dyn Array>> =
                    Vec::with_capacity(filter.column_indices().len());
                for column_index in filter.column_indices() {
                    let array = if column_index.side == JoinSide::Left {
                        Arc::clone(left_gathered.column(column_index.index))
                    } else {
                        let col = right_batch.column(column_index.index);
                        take(col.as_ref(), &right_indices, None)?
                    };
                    filter_columns.push(array);
                }

                RecordBatch::try_new(Arc::new((*filter.schema).clone()), filter_columns)?
            };

            let filter_result = filter
                .expression()
                .evaluate(&intermediate_batch)?
                .into_array(intermediate_batch.num_rows())?;
            let filter_arr = as_boolean_array(&filter_result)?;

            // Combine with null bitmap to get a unified mask
            boolean_mask_from_filter(filter_arr)
        } else {
            // No filter: all pairs match
            BooleanArray::from(vec![true; total_rows])
        };

        // Update the global left or right bitmap for matched indices
        // -----------------------------------------------------------

        // None means we don't have to update left bitmap for this join type
        let mut left_bitmap = if need_produce_result_in_final(self.join_type) {
            Some(left_data.bitmap().lock())
        } else {
            None
        };

        // 'local' meaning: we want to collect 'is_matched' flag for the current
        // right batch, after it has joining all of the left buffer, here it's only
        // the partial result for joining given left range
        let mut local_right_bitmap = if self.should_track_unmatched_right {
            let mut current_right_batch_bitmap = BooleanBufferBuilder::new(right_rows);
            // Ensure builder has logical length so set_bit is in-bounds
            current_right_batch_bitmap.append_n(right_rows, false);
            Some(current_right_batch_bitmap)
        } else {
            None
        };

        // Set the matched bit for left and right side bitmap
        for (i, is_matched) in bitmap_combined.iter().enumerate() {
            let is_matched = is_matched.ok_or_else(|| {
                internal_datafusion_err!("Must be Some after the previous combining step")
            })?;

            let l_index = l_start_index + i / right_rows;
            let r_index = i % right_rows;

            if let Some(bitmap) = left_bitmap.as_mut()
                && is_matched
            {
                // Map local index back to absolute left index within the batch
                bitmap.set_bit(l_index, true);
            }

            if let Some(bitmap) = local_right_bitmap.as_mut()
                && is_matched
            {
                bitmap.set_bit(r_index, true);
            }
        }

        // Apply the local right bitmap to the global bitmap
        if self.should_track_unmatched_right {
            // Remember to put it back after update
            let global_right_bitmap =
                std::mem::take(&mut self.current_right_batch_matched).ok_or_else(
                    || internal_datafusion_err!("right batch's bitmap should be present"),
                )?;
            let (buf, nulls) = global_right_bitmap.into_parts();
            debug_assert!(nulls.is_none());

            let current_right_bitmap = local_right_bitmap
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "Should be Some if the current join type requires right bitmap"
                    )
                })?
                .finish();
            let updated_global_right_bitmap = buf.bitor(&current_right_bitmap);

            self.current_right_batch_matched =
                Some(BooleanArray::new(updated_global_right_bitmap, None));
        }

        // For the following join types: only bitmaps are updated; do not emit rows now
        if matches!(
            self.join_type,
            JoinType::LeftAnti
                | JoinType::LeftSemi
                | JoinType::LeftMark
                | JoinType::RightAnti
                | JoinType::RightMark
                | JoinType::RightSemi
        ) {
            return Ok(None);
        }

        // Build the projected output batch (using output schema/column_indices),
        // then apply the bitmap filter to it.
        if self.output_schema.fields().is_empty() {
            // Empty projection: only row count matters
            let row_count = bitmap_combined.true_count();
            return Ok(Some(create_record_batch_with_empty_schema(
                Arc::clone(&self.output_schema),
                row_count,
            )?));
        }

        let mut out_columns: Vec<Arc<dyn Array>> =
            Vec::with_capacity(self.output_schema.fields().len());
        for column_index in &self.column_indices {
            let array = if column_index.side == JoinSide::Left {
                Arc::clone(left_gathered.column(column_index.index))
            } else {
                let col = right_batch.column(column_index.index);
                take(col.as_ref(), &right_indices, None)?
            };
            out_columns.push(array);
        }
        let pre_filtered =
            RecordBatch::try_new(Arc::clone(&self.output_schema), out_columns)?;
        let filtered = filter_record_batch(&pre_filtered, &bitmap_combined)?;
        Ok(Some(filtered))
    }

    /// Process a single left row join with the current right batch.
    /// Returns the join results, split into multiple batches if a large
    /// build-side value has to be broadcast across many probe rows (see
    /// [`build_row_join_batch`]). Empty if there is nothing to output.
    ///
    /// Side Effect: If the join type requires, left or right side matched bitmap
    /// will be set for matched indices.
    fn process_single_left_row_join(
        &mut self,
        left_data: &JoinLeftData,
        right_batch: &RecordBatch,
        l_index: usize,
    ) -> Result<Vec<RecordBatch>> {
        let right_row_count = right_batch.num_rows();
        if right_row_count == 0 {
            return Ok(vec![]);
        }

        // Resolve the flat left-row index to the batch it lives in and the row
        // offset within that batch (the build side is not concatenated).
        let (left_batch_idx, left_row_idx) = left_data.locate(l_index);
        let left_batch = left_data.batch_at(left_batch_idx);

        let cur_right_bitmap = if let Some(filter) = &self.join_filter {
            apply_filter_to_row_join_batch(left_batch, left_row_idx, right_batch, filter)?
        } else {
            BooleanArray::from(vec![true; right_row_count])
        };

        self.update_matched_bitmap(l_index, &cur_right_bitmap)?;

        // For the following join types: here we only have to set the left/right
        // bitmap, and no need to output result
        if matches!(
            self.join_type,
            JoinType::LeftAnti
                | JoinType::LeftSemi
                | JoinType::LeftMark
                | JoinType::RightAnti
                | JoinType::RightMark
                | JoinType::RightSemi
        ) {
            return Ok(vec![]);
        }

        if !cur_right_bitmap.has_true() {
            // If none of the pairs has passed the join predicate/filter
            Ok(vec![])
        } else {
            // Use the optimized approach similar to build_intermediate_batch_for_single_left_row
            build_row_join_batch(
                &self.output_schema,
                left_batch,
                left_row_idx,
                right_batch,
                Some(cur_right_bitmap),
                &self.column_indices,
                JoinSide::Left,
            )
        }
    }

    /// Returns bool to indicate should it continue processing unmatched rows
    /// true -> continue in the same EmitLeftUnmatched state
    /// false -> next state (Done)
    fn process_left_unmatched(&mut self) -> Result<bool> {
        let left_data = self.get_left_data()?;
        let num_rows = left_data.num_rows();

        // ========
        // Check early return conditions
        // ========

        // Early return if join type can't have unmatched rows
        let join_type_no_produce_left = !need_produce_result_in_final(self.join_type);
        // Early return if another thread is already processing unmatched rows
        let handled_by_other_partition =
            self.left_emit_idx == 0 && !left_data.report_probe_completed();
        // Stop processing unmatched rows, the caller will go to the next state
        let finished = self.left_emit_idx >= num_rows;

        if join_type_no_produce_left || handled_by_other_partition || finished {
            return Ok(false);
        }

        // ========
        // Process unmatched rows and push the result into output_buffer
        // Each time, the number to process is up to batch size. Because the build
        // side is stored as separate batches, also clamp the range to the batch
        // containing `start_idx` so the slice stays within a single batch.
        // ========
        let start_idx = self.left_emit_idx;
        let (start_batch_idx, _) = left_data.locate(start_idx);
        let batch_end = left_data.offsets()[start_batch_idx + 1];
        let end_idx = std::cmp::min(start_idx + self.batch_size, batch_end);

        if let Some(batch) =
            self.process_left_unmatched_range(left_data, start_idx, end_idx)?
        {
            self.push_output_batch(batch)?;
        }

        // ==== Prepare for the next iteration ====
        self.left_emit_idx = end_idx;

        // Return true to continue processing unmatched rows
        Ok(true)
    }

    /// Process unmatched rows from the left data within the specified range.
    /// Returns a RecordBatch containing the unmatched rows (None if empty).
    ///
    /// # Arguments
    /// * `left_data` - The left side data containing the batch and bitmap
    /// * `start_idx` - Start index (inclusive) of the range to process
    /// * `end_idx` - End index (exclusive) of the range to process
    ///
    /// # Safety
    /// The caller is responsible for ensuring that `start_idx` and `end_idx` are
    /// within valid bounds of the left batch. This function does not perform
    /// bounds checking.
    fn process_left_unmatched_range(
        &self,
        left_data: &JoinLeftData,
        start_idx: usize,
        end_idx: usize,
    ) -> Result<Option<RecordBatch>> {
        if start_idx == end_idx {
            return Ok(None);
        }

        // Slice both left batch, and bitmap to range [start_idx, end_idx)
        // The range is bit index (not byte). The caller guarantees the range lies
        // within a single build batch, so resolve `start_idx` to that batch and
        // slice it directly (no concatenation of the build side).
        let (batch_idx, local_start) = left_data.locate(start_idx);
        let left_batch_sliced = left_data
            .batch_at(batch_idx)
            .slice(local_start, end_idx - start_idx);

        // Can this be more efficient?
        let mut bitmap_sliced = BooleanBufferBuilder::new(end_idx - start_idx);
        bitmap_sliced.append_n(end_idx - start_idx, false);
        let bitmap = left_data.bitmap().lock();
        for i in start_idx..end_idx {
            assert!(
                i - start_idx < bitmap_sliced.capacity(),
                "DBG: {start_idx}, {end_idx}"
            );
            bitmap_sliced.set_bit(i - start_idx, bitmap.get_bit(i));
        }
        let bitmap_sliced = BooleanArray::new(bitmap_sliced.finish(), None);

        let right_schema = self
            .right_data
            .as_ref()
            .expect("right_data must be present when building unmatched batch")
            .schema();
        build_unmatched_batch(
            &self.output_schema,
            &left_batch_sliced,
            bitmap_sliced,
            &right_schema,
            &self.column_indices,
            self.join_type,
            JoinSide::Left,
        )
    }

    /// Process unmatched rows from the current right batch and reset the bitmap.
    /// Returns a RecordBatch containing the unmatched right rows (None if empty).
    fn process_right_unmatched(&mut self) -> Result<Option<RecordBatch>> {
        // ==== Take current right batch and its bitmap ====
        let right_batch_bitmap: BooleanArray =
            std::mem::take(&mut self.current_right_batch_matched).ok_or_else(|| {
                internal_datafusion_err!("right bitmap should be available")
            })?;

        let right_batch = self.current_right_batch.take();
        let cur_right_batch = unwrap_or_internal_err!(right_batch);

        let left_data = self.get_left_data()?;
        let left_schema = Arc::clone(left_data.schema());

        let res = build_unmatched_batch(
            &self.output_schema,
            &cur_right_batch,
            right_batch_bitmap,
            &left_schema,
            &self.column_indices,
            self.join_type,
            JoinSide::Right,
        );

        // ==== Clean-up ====
        self.current_right_batch_matched = None;

        res
    }

    // ==== Utilities ====

    /// Get the build-side data of the left input, errors if it's None
    fn get_left_data(&self) -> Result<&Arc<JoinLeftData>> {
        self.buffered_left_data
            .as_ref()
            .ok_or_else(|| internal_datafusion_err!("LeftData should be available"))
    }

    /// Flush the `output_buffer` if there are batches ready to output
    /// None if no result batch ready.
    fn maybe_flush_ready_batch(&mut self) -> Option<Poll<Option<Result<RecordBatch>>>> {
        if self.output_buffer.has_completed_batch()
            && let Some(batch) = self.output_buffer.next_completed_batch()
        {
            // Update output rows for selectivity metric
            let output_rows = batch.num_rows();
            self.metrics.selectivity.add_part(output_rows);

            return Some(Poll::Ready(Some(Ok(batch))));
        }

        None
    }

    /// Push a batch into `output_buffer`, force-finishing the buffer first if
    /// appending the batch could overflow `i32` offsets when the buffered rows
    /// are later concatenated into a single completed batch.
    ///
    /// [`BatchCoalescer`] concatenates buffered batches to complete an output
    /// batch, and that concatenation fails if a `Utf8`/`Binary`/`List` column
    /// accumulates more than `i32::MAX` bytes. Chunked broadcast batches from
    /// [`build_row_join_batch`] can individually approach that limit, so
    /// without this check the coalescer would merge them back over it.
    ///
    /// All `output_buffer.push_batch()` calls must go through here.
    fn push_output_batch(&mut self, batch: RecordBatch) -> Result<()> {
        if !self.var_len_output_columns.is_empty() {
            // If the size of a column can't be computed (not expected for the
            // tracked types), fall back to usize::MAX: the buffer is then
            // force-finished around every push, which disables coalescing but
            // stays correct.
            let incoming: Vec<usize> = self
                .var_len_output_columns
                .iter()
                .map(|&idx| {
                    batch
                        .column(idx)
                        .to_data()
                        .get_slice_memory_size()
                        .unwrap_or(usize::MAX)
                })
                .collect();
            let would_overflow = self
                .buffered_var_len_bytes
                .iter()
                .zip(&incoming)
                .any(|(cur, inc)| cur.saturating_add(*inc) > MAX_BATCH_VAR_BYTES);
            if would_overflow {
                self.finish_output_buffer()?;
            }
            for (cur, inc) in self.buffered_var_len_bytes.iter_mut().zip(&incoming) {
                *cur = cur.saturating_add(*inc);
            }
        }
        self.output_buffer.push_batch(batch)?;
        Ok(())
    }

    /// Force-complete `output_buffer`'s partially buffered rows and reset the
    /// variable-length byte counters. All `output_buffer.finish_buffered_batch()`
    /// calls must go through here so the counters stay in sync.
    fn finish_output_buffer(&mut self) -> Result<()> {
        self.output_buffer.finish_buffered_batch()?;
        self.buffered_var_len_bytes.fill(0);
        Ok(())
    }

    /// After joining (l_index@left_buffer x current_right_batch), it will result
    /// in a bitmap (the same length as current_right_batch) as the join match
    /// result. Use this bitmap to update the global bitmap, for special join
    /// types like full joins.
    ///
    /// Example:
    /// After joining l_index=1 (1-indexed row in the left buffer), and the
    /// current right batch with 3 elements, this function will be called with
    /// arguments: l_index = 1, r_matched = [false, false, true]
    /// - If the join type is FullJoin, the 1-index in the left bitmap will be
    ///   set to true, and also the right bitmap will be bitwise-ORed with the
    ///   input r_matched bitmap.
    /// - For join types that don't require output unmatched rows, this
    ///   function can be a no-op. For inner joins, this function is a no-op; for left
    ///   joins, only the left bitmap may be updated.
    fn update_matched_bitmap(
        &mut self,
        l_index: usize,
        r_matched_bitmap: &BooleanArray,
    ) -> Result<()> {
        let left_data = self.get_left_data()?;

        // 1. Maybe update the left bitmap
        if need_produce_result_in_final(self.join_type) && r_matched_bitmap.has_true() {
            let mut bitmap = left_data.bitmap().lock();
            bitmap.set_bit(l_index, true);
        }

        // 2. Maybe update the right bitmap
        if self.should_track_unmatched_right {
            debug_assert!(self.current_right_batch_matched.is_some());
            // after bit-wise or, it will be put back
            let right_bitmap = std::mem::take(&mut self.current_right_batch_matched)
                .ok_or_else(|| {
                    internal_datafusion_err!("right batch's bitmap should be present")
                })?;
            let (buf, nulls) = right_bitmap.into_parts();
            debug_assert!(nulls.is_none());
            let updated_right_bitmap = buf.bitor(r_matched_bitmap.values());

            self.current_right_batch_matched =
                Some(BooleanArray::new(updated_right_bitmap, None));
        }

        Ok(())
    }
}

// ==== Utilities ====

/// Apply the join filter between:
/// (l_index th row in left buffer) x (right batch)
/// Returns a bitmap, with successfully joined indices set to true
fn apply_filter_to_row_join_batch(
    left_batch: &RecordBatch,
    l_index: usize,
    right_batch: &RecordBatch,
    filter: &JoinFilter,
) -> Result<BooleanArray> {
    apply_filter_to_row_join_batch_with_limit(
        left_batch,
        l_index,
        right_batch,
        filter,
        MAX_BATCH_VAR_BYTES,
    )
}

/// Implementation of [`apply_filter_to_row_join_batch`] with an injectable
/// broadcast byte limit (see [`build_row_join_batch_with_limit`]).
fn apply_filter_to_row_join_batch_with_limit(
    left_batch: &RecordBatch,
    l_index: usize,
    right_batch: &RecordBatch,
    filter: &JoinFilter,
    broadcast_byte_limit: usize,
) -> Result<BooleanArray> {
    debug_assert!(left_batch.num_rows() != 0 && right_batch.num_rows() != 0);

    let intermediate_batches = if filter.schema.fields().is_empty() {
        // If filter is constant (e.g. literal `true`), empty batch can be used
        // in the later filter step.
        vec![create_record_batch_with_empty_schema(
            Arc::new((*filter.schema).clone()),
            right_batch.num_rows(),
        )?]
    } else {
        // The intermediate batch is split into chunks if broadcasting a large
        // left value across the right batch would overflow `i32` offsets. The
        // chunks partition the right batch in order, so the per-chunk filter
        // masks concatenate back into a mask for the whole right batch.
        build_row_join_batch_with_limit(
            &filter.schema,
            left_batch,
            l_index,
            right_batch,
            None,
            &filter.column_indices,
            JoinSide::Left,
            broadcast_byte_limit,
        )?
    };

    if intermediate_batches.is_empty() {
        return internal_err!(
            "This function assume input batch is not empty, so the intermediate batch can't be empty too"
        );
    }

    let mut masks = Vec::with_capacity(intermediate_batches.len());
    for intermediate_batch in intermediate_batches {
        let filter_result = filter
            .expression()
            .evaluate(&intermediate_batch)?
            .into_array(intermediate_batch.num_rows())?;
        let filter_arr = as_boolean_array(&filter_result)?;

        // Convert boolean array with potential nulls into a unified mask bitmap
        masks.push(boolean_mask_from_filter(filter_arr));
    }

    if masks.len() == 1 {
        Ok(masks.swap_remove(0))
    } else {
        let mask_refs: Vec<&dyn Array> =
            masks.iter().map(|mask| mask as &dyn Array).collect();
        Ok(as_boolean_array(&concat(&mask_refs)?)?.clone())
    }
}

/// Convert a boolean filter array into a unified mask bitmap.
///
/// Caution: The filter result is NOT a bitmap; it contains true/false/null values.
/// For example, `1 < NULL` evaluates to NULL. Therefore, we must combine (AND)
/// the boolean array with its null bitmap to construct a unified bitmap.
#[inline]
fn boolean_mask_from_filter(filter_arr: &BooleanArray) -> BooleanArray {
    let (values, nulls) = filter_arr.clone().into_parts();
    match nulls {
        Some(nulls) => BooleanArray::new(nulls.inner() & &values, None),
        None => BooleanArray::new(values, None),
    }
}

/// This function performs the following steps:
/// 1. Apply filter to probe-side batch
/// 2. Broadcast the left row (build_side_batch\[build_side_index\]) to the
///    filtered probe-side batch
/// 3. Concat them together according to `col_indices`, and return the result
///    (None if the result is empty)
///
/// Example:
/// build_side_batch:
/// a
/// ----
/// 1
/// 2
/// 3
///
/// # 0 index element in the build_side_batch (that is `1`) will be used
/// build_side_index: 0
///
/// probe_side_batch:
/// b
/// ----
/// 10
/// 20
/// 30
/// 40
///
/// # After applying it, only index 1 and 3 elements in probe_side_batch will be
/// # kept
/// probe_side_filter:
/// false
/// true
/// false
/// true
///
///
/// # Projections to the build/probe side batch, to construct the output batch
/// col_indices:
/// [(left, 0), (right, 0)]
///
/// build_side: left
///
/// ====
/// Result batch:
/// a b
/// ----
/// 1 20
/// 1 40
/// `Utf8`/`Binary` arrays use `i32` offsets, so a single array's values buffer
/// cannot hold more than `i32::MAX` bytes. Broadcasting one build-side value
/// across `n` output rows materializes `value_len * n` bytes, which must stay
/// under this limit (`OffsetBuffer::from_repeated_length` panics otherwise).
/// The same limit applies when [`BatchCoalescer`] concatenates buffered
/// batches into one completed output batch.
const MAX_BATCH_VAR_BYTES: usize = i32::MAX as usize;

/// Returns true if the data type (or any nested child type) stores
/// variable-length data behind `i32` offsets (`Utf8`, `Binary`, `List`, `Map`),
/// i.e. types for which a single array is capped at `i32::MAX` bytes/elements
/// and can therefore overflow when many rows are materialized into one array.
fn contains_i32_offset_data(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8 | DataType::Binary | DataType::List(_) | DataType::Map(_, _) => {
            true
        }
        DataType::LargeList(field) | DataType::FixedSizeList(field, _) => {
            contains_i32_offset_data(field.data_type())
        }
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_i32_offset_data(field.data_type())),
        DataType::Dictionary(_, value_type) => contains_i32_offset_data(value_type),
        DataType::RunEndEncoded(_, values) => {
            contains_i32_offset_data(values.data_type())
        }
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| contains_i32_offset_data(field.data_type())),
        _ => false,
    }
}

/// Compute the maximum number of rows one output batch may contain when
/// broadcasting the `build_side_index`-th build row, such that no
/// `Utf8`/`Binary` build column's repeated value exceeds `byte_limit` total
/// bytes (`i32` offset overflow otherwise).
///
/// Other build column types either have no `i32` offsets (primitives, views,
/// `Large*`), or are broadcast via `take`, which returns a proper error
/// instead of panicking on offset overflow.
fn max_broadcast_rows(
    build_side_batch: &RecordBatch,
    build_side_index: usize,
    col_indices: &[ColumnIndex],
    build_side: JoinSide,
    byte_limit: usize,
) -> usize {
    let mut max_rows = usize::MAX;
    for column_index in col_indices {
        if column_index.side != build_side {
            continue;
        }
        let array = build_side_batch.column(column_index.index);
        if array.is_null(build_side_index) {
            continue;
        }
        let value_len = match array.data_type() {
            DataType::Utf8 => array.as_string::<i32>().value(build_side_index).len(),
            DataType::Binary => array.as_binary::<i32>().value(build_side_index).len(),
            _ => continue,
        };
        if value_len == 0 {
            continue;
        }
        max_rows = max_rows.min(byte_limit / value_len);
    }
    // A single row is always representable: the value comes from an existing
    // i32-offset array, so its length fits in an i32.
    max_rows.max(1)
}

/// See [`build_row_join_batch_with_limit`]. This wrapper applies the real
/// `i32` offset limit.
fn build_row_join_batch(
    output_schema: &Schema,
    build_side_batch: &RecordBatch,
    build_side_index: usize,
    probe_side_batch: &RecordBatch,
    probe_side_filter: Option<BooleanArray>,
    col_indices: &[ColumnIndex],
    build_side: JoinSide,
) -> Result<Vec<RecordBatch>> {
    build_row_join_batch_with_limit(
        output_schema,
        build_side_batch,
        build_side_index,
        probe_side_batch,
        probe_side_filter,
        col_indices,
        build_side,
        MAX_BATCH_VAR_BYTES,
    )
}

/// Join a single build-side row with the (filtered) probe-side batch, by
/// broadcasting the build row across the probe rows.
///
/// The result is split into multiple batches when broadcasting a large
/// `Utf8`/`Binary` build value would otherwise overflow the array's `i32`
/// offsets (each batch's repeated value stays within `broadcast_byte_limit`
/// total bytes). In the common case of small values a single batch is
/// returned; an empty `Vec` means there is nothing to output.
///
/// `broadcast_byte_limit` is injectable so tests can exercise the chunking
/// without allocating gigabytes; production code uses [`build_row_join_batch`].
#[expect(clippy::too_many_arguments)]
fn build_row_join_batch_with_limit(
    output_schema: &Schema,
    build_side_batch: &RecordBatch,
    build_side_index: usize,
    probe_side_batch: &RecordBatch,
    probe_side_filter: Option<BooleanArray>,
    // See [`NLJStream`] struct's `column_indices` field for more detail
    col_indices: &[ColumnIndex],
    // If the build side is left or right, used to interpret the side information
    // in `col_indices`
    build_side: JoinSide,
    broadcast_byte_limit: usize,
) -> Result<Vec<RecordBatch>> {
    debug_assert!(build_side != JoinSide::None);

    // TODO(perf): since the output might be projection of right batch, this
    // filtering step is more efficient to be done inside the column_index loop
    let filtered_probe_batch = if let Some(filter) = probe_side_filter {
        &filter_record_batch(probe_side_batch, &filter)?
    } else {
        probe_side_batch
    };

    if filtered_probe_batch.num_rows() == 0 {
        return Ok(vec![]);
    }

    // Edge case: downstream operator does not require any columns from this NLJ,
    // so allow an empty projection.
    // Example:
    //  SELECT DISTINCT 32 AS col2
    //  FROM tab0 AS cor0
    //  LEFT OUTER JOIN tab2 AS cor1
    //  ON ( NULL ) IS NULL;
    if output_schema.fields.is_empty() {
        return Ok(vec![create_record_batch_with_empty_schema(
            Arc::new(output_schema.clone()),
            filtered_probe_batch.num_rows(),
        )?]);
    }

    let num_rows = filtered_probe_batch.num_rows();
    let chunk_rows = max_broadcast_rows(
        build_side_batch,
        build_side_index,
        col_indices,
        build_side,
        broadcast_byte_limit,
    );

    // Common case: all build values are small enough to broadcast across the
    // whole probe batch in one output batch
    if chunk_rows >= num_rows {
        return Ok(vec![broadcast_build_row(
            output_schema,
            build_side_batch,
            build_side_index,
            filtered_probe_batch,
            col_indices,
            build_side,
        )?]);
    }

    let mut batches = Vec::with_capacity(num_rows.div_ceil(chunk_rows));
    let mut offset = 0;
    while offset < num_rows {
        let len = chunk_rows.min(num_rows - offset);
        let probe_chunk = filtered_probe_batch.slice(offset, len);
        batches.push(broadcast_build_row(
            output_schema,
            build_side_batch,
            build_side_index,
            &probe_chunk,
            col_indices,
            build_side,
        )?);
        offset += len;
    }
    Ok(batches)
}

/// Build one output batch pairing the `build_side_index`-th build row with
/// every row of `probe_batch`: build-side columns are broadcast to the probe
/// batch length, probe-side columns are passed through.
///
/// The caller is responsible for keeping `probe_batch` small enough that the
/// broadcast `Utf8`/`Binary` columns do not overflow `i32` offsets (see
/// [`max_broadcast_rows`]).
fn broadcast_build_row(
    output_schema: &Schema,
    build_side_batch: &RecordBatch,
    build_side_index: usize,
    probe_batch: &RecordBatch,
    col_indices: &[ColumnIndex],
    build_side: JoinSide,
) -> Result<RecordBatch> {
    let mut columns: Vec<Arc<dyn Array>> =
        Vec::with_capacity(output_schema.fields().len());

    for column_index in col_indices {
        let array = if column_index.side == build_side {
            // Broadcast the single build-side row to match the probe-side
            // batch length
            let original_left_array = build_side_batch.column(column_index.index);

            // Use `arrow::compute::take` directly for `List(Utf8View)` rather
            // than going through `ScalarValue::to_array_of_size()`, which
            // avoids some intermediate allocations.
            //
            // In other cases, `to_array_of_size()` is faster.
            match original_left_array.data_type() {
                DataType::List(field) | DataType::LargeList(field)
                    if field.data_type() == &DataType::Utf8View =>
                {
                    let indices_iter = std::iter::repeat_n(
                        build_side_index as u64,
                        probe_batch.num_rows(),
                    );
                    let indices_array = UInt64Array::from_iter_values(indices_iter);
                    take(original_left_array.as_ref(), &indices_array, None)?
                }
                _ => {
                    let scalar_value = ScalarValue::try_from_array(
                        original_left_array.as_ref(),
                        build_side_index,
                    )?;
                    scalar_value.to_array_of_size(probe_batch.num_rows())?
                }
            }
        } else {
            // Take the probe-side column as is
            Arc::clone(probe_batch.column(column_index.index))
        };

        columns.push(array);
    }

    Ok(RecordBatch::try_new(
        Arc::new(output_schema.clone()),
        columns,
    )?)
}

/// Special case for `PlaceHolderRowExec`
/// Minimal example:  SELECT 1 WHERE EXISTS (SELECT 1);
//
/// # Return
/// If Some, that's the result batch
/// If None, it's not for this special case. Continue execution.
fn build_unmatched_batch_empty_schema(
    output_schema: &SchemaRef,
    batch_bitmap: &BooleanArray,
    // For left/right/full joins, it needs to fill nulls for another side
    join_type: JoinType,
) -> Result<Option<RecordBatch>> {
    let result_size = match join_type {
        JoinType::Left
        | JoinType::Right
        | JoinType::Full
        | JoinType::LeftAnti
        | JoinType::RightAnti => batch_bitmap.false_count(),
        JoinType::LeftSemi | JoinType::RightSemi => batch_bitmap.true_count(),
        JoinType::LeftMark | JoinType::RightMark => batch_bitmap.len(),
        _ => unreachable!(),
    };

    if output_schema.fields().is_empty() {
        Ok(Some(create_record_batch_with_empty_schema(
            Arc::clone(output_schema),
            result_size,
        )?))
    } else {
        Ok(None)
    }
}

/// Creates an empty RecordBatch with a specific row count.
/// This is useful for cases where we need a batch with the correct schema and row count
/// but no actual data columns (e.g., for constant filters).
fn create_record_batch_with_empty_schema(
    schema: SchemaRef,
    row_count: usize,
) -> Result<RecordBatch> {
    let options = RecordBatchOptions::new()
        .with_match_field_names(true)
        .with_row_count(Some(row_count));

    RecordBatch::try_new_with_options(schema, vec![], &options).map_err(|e| {
        internal_datafusion_err!("Failed to create empty record batch: {}", e)
    })
}

/// # Example:
/// batch:
/// a
/// ----
/// 1
/// 2
/// 3
///
/// batch_bitmap:
/// ----
/// false
/// true
/// false
///
/// another_side_schema:
/// [(b, bool), (c, int32)]
///
/// join_type: JoinType::Left
///
/// col_indices: ...(please refer to the comment in `NLJStream::column_indices``)
///
/// batch_side: right
///
/// # Walkthrough:
///
/// This executor is performing a right join, and the currently processed right
/// batch is as above. After joining it with all buffered left rows, the joined
/// entries are marked by the `batch_bitmap`.
/// This method will keep the unmatched indices on the batch side (right), and pad
/// the left side with nulls. The result would be:
///
/// b          c           a
/// ------------------------
/// Null(bool) Null(Int32) 1
/// Null(bool) Null(Int32) 3
fn build_unmatched_batch(
    output_schema: &SchemaRef,
    batch: &RecordBatch,
    batch_bitmap: BooleanArray,
    // For left/right/full joins, it needs to fill nulls for another side
    another_side_schema: &SchemaRef,
    col_indices: &[ColumnIndex],
    join_type: JoinType,
    batch_side: JoinSide,
) -> Result<Option<RecordBatch>> {
    // Should not call it for inner joins
    debug_assert_ne!(join_type, JoinType::Inner);
    debug_assert_ne!(batch_side, JoinSide::None);

    // Handle special case (see function comment)
    if let Some(batch) =
        build_unmatched_batch_empty_schema(output_schema, &batch_bitmap, join_type)?
    {
        return Ok(Some(batch));
    }

    match join_type {
        JoinType::Full | JoinType::Right | JoinType::Left => {
            if join_type == JoinType::Right {
                debug_assert_eq!(batch_side, JoinSide::Right);
            }
            if join_type == JoinType::Left {
                debug_assert_eq!(batch_side, JoinSide::Left);
            }

            // 1. Filter the batch with *flipped* bitmap
            // 2. Fill left side with nulls
            let flipped_bitmap = not(&batch_bitmap)?;

            // create a record batch, with left_schema, of only one row of all nulls
            let left_null_columns: Vec<Arc<dyn Array>> = another_side_schema
                .fields()
                .iter()
                .map(|field| new_null_array(field.data_type(), 1))
                .collect();

            // Hack: If the left schema is not nullable, the full join result
            // might contain null, this is only a temporary batch to construct
            // such full join result.
            let nullable_left_schema = Arc::new(Schema::new(
                another_side_schema
                    .fields()
                    .iter()
                    .map(|field| (**field).clone().with_nullable(true))
                    .collect::<Vec<_>>(),
            ));
            let left_null_batch = if nullable_left_schema.fields.is_empty() {
                // Left input can be an empty relation, in this case left relation
                // won't be used to construct the result batch (i.e. not in `col_indices`)
                create_record_batch_with_empty_schema(nullable_left_schema, 0)?
            } else {
                RecordBatch::try_new(nullable_left_schema, left_null_columns)?
            };

            debug_assert_ne!(batch_side, JoinSide::None);
            let opposite_side = batch_side.negate();

            // The broadcast row is all nulls (zero variable-length bytes), so
            // `build_row_join_batch` never splits the result into chunks.
            let mut batches = build_row_join_batch(
                output_schema,
                &left_null_batch,
                0,
                batch,
                Some(flipped_bitmap),
                col_indices,
                opposite_side,
            )?;
            if batches.len() > 1 {
                return internal_err!(
                    "broadcasting an all-null row must produce at most one batch"
                );
            }
            Ok(batches.pop())
        }
        JoinType::RightSemi
        | JoinType::RightAnti
        | JoinType::LeftSemi
        | JoinType::LeftAnti => {
            if matches!(join_type, JoinType::RightSemi | JoinType::RightAnti) {
                debug_assert_eq!(batch_side, JoinSide::Right);
            }
            if matches!(join_type, JoinType::LeftSemi | JoinType::LeftAnti) {
                debug_assert_eq!(batch_side, JoinSide::Left);
            }

            let bitmap = if matches!(join_type, JoinType::LeftSemi | JoinType::RightSemi)
            {
                batch_bitmap.clone()
            } else {
                not(&batch_bitmap)?
            };

            if !bitmap.has_true() {
                return Ok(None);
            }

            let mut columns: Vec<Arc<dyn Array>> =
                Vec::with_capacity(output_schema.fields().len());

            for column_index in col_indices {
                debug_assert!(column_index.side == batch_side);

                let col = batch.column(column_index.index);
                let filtered_col = filter(col, &bitmap)?;

                columns.push(filtered_col);
            }

            Ok(Some(RecordBatch::try_new(
                Arc::clone(output_schema),
                columns,
            )?))
        }
        JoinType::RightMark | JoinType::LeftMark => {
            if join_type == JoinType::RightMark {
                debug_assert_eq!(batch_side, JoinSide::Right);
            }
            if join_type == JoinType::LeftMark {
                debug_assert_eq!(batch_side, JoinSide::Left);
            }

            let mut columns: Vec<Arc<dyn Array>> =
                Vec::with_capacity(output_schema.fields().len());

            // Hack to deal with the borrow checker
            let mut right_batch_bitmap_opt = Some(batch_bitmap);

            for column_index in col_indices {
                if column_index.side == batch_side {
                    let col = batch.column(column_index.index);

                    columns.push(Arc::clone(col));
                } else if column_index.side == JoinSide::None {
                    let right_batch_bitmap = std::mem::take(&mut right_batch_bitmap_opt);
                    match right_batch_bitmap {
                        Some(right_batch_bitmap) => {
                            columns.push(Arc::new(right_batch_bitmap))
                        }
                        None => unreachable!("Should only be one mark column"),
                    }
                } else {
                    return internal_err!(
                        "Not possible to have this join side for RightMark join"
                    );
                }
            }

            Ok(Some(RecordBatch::try_new(
                Arc::clone(output_schema),
                columns,
            )?))
        }
        _ => internal_err!(
            "If batch is at right side, this function must be handling Full/Right/RightSemi/RightAnti/RightMark joins"
        ),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::test::{TestMemoryExec, assert_join_metrics};
    use crate::{
        common, expressions::Column, repartition::RepartitionExec, test::build_table_i32,
    };

    use arrow::compute::SortOptions;
    use arrow::datatypes::{DataType, Field};
    use datafusion_common::assert_contains;
    use datafusion_common::test_util::batches_to_sort_string;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, Literal};
    use datafusion_physical_expr::{Partitioning, PhysicalExpr};
    use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};

    use insta::allow_duplicates;
    use insta::assert_snapshot;
    use rstest::rstest;

    fn build_table(
        a: (&str, &Vec<i32>),
        b: (&str, &Vec<i32>),
        c: (&str, &Vec<i32>),
        batch_size: Option<usize>,
        sorted_column_names: Vec<&str>,
    ) -> Arc<dyn ExecutionPlan> {
        let batch = build_table_i32(a, b, c);
        let schema = batch.schema();

        let batches = if let Some(batch_size) = batch_size {
            let num_batches = batch.num_rows().div_ceil(batch_size);
            (0..num_batches)
                .map(|i| {
                    let start = i * batch_size;
                    let remaining_rows = batch.num_rows() - start;
                    batch.slice(start, batch_size.min(remaining_rows))
                })
                .collect::<Vec<_>>()
        } else {
            vec![batch]
        };

        let mut sort_info = vec![];
        for name in sorted_column_names {
            let index = schema.index_of(name).unwrap();
            let sort_expr = PhysicalSortExpr::new(
                Arc::new(Column::new(name, index)),
                SortOptions::new(false, false),
            );
            sort_info.push(sort_expr);
        }
        let mut source = TestMemoryExec::try_new(&[batches], schema, None).unwrap();
        if let Some(ordering) = LexOrdering::new(sort_info) {
            source = source.try_with_sort_information(vec![ordering]).unwrap();
        }

        let source = Arc::new(source);
        Arc::new(TestMemoryExec::update_cache(&source))
    }

    fn build_left_table() -> Arc<dyn ExecutionPlan> {
        build_table(
            ("a1", &vec![5, 9, 11]),
            ("b1", &vec![5, 8, 8]),
            ("c1", &vec![50, 90, 110]),
            None,
            Vec::new(),
        )
    }

    fn build_right_table() -> Arc<dyn ExecutionPlan> {
        build_table(
            ("a2", &vec![12, 2, 10]),
            ("b2", &vec![10, 2, 10]),
            ("c2", &vec![40, 80, 100]),
            None,
            Vec::new(),
        )
    }

    fn prepare_join_filter() -> JoinFilter {
        let column_indices = vec![
            ColumnIndex {
                index: 1,
                side: JoinSide::Left,
            },
            ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            },
        ];
        let intermediate_schema = Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("x", DataType::Int32, true),
        ]);
        // left.b1!=8
        let left_filter = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(8)))),
        )) as Arc<dyn PhysicalExpr>;
        // right.b2!=10
        let right_filter = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 1)),
            Operator::NotEq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
        )) as Arc<dyn PhysicalExpr>;
        // filter = left.b1!=8 and right.b2!=10
        // after filter:
        // left table:
        // ("a1", &vec![5]),
        // ("b1", &vec![5]),
        // ("c1", &vec![50]),
        // right table:
        // ("a2", &vec![12, 2]),
        // ("b2", &vec![10, 2]),
        // ("c2", &vec![40, 80]),
        let filter_expression =
            Arc::new(BinaryExpr::new(left_filter, Operator::And, right_filter))
                as Arc<dyn PhysicalExpr>;

        JoinFilter::new(
            filter_expression,
            column_indices,
            Arc::new(intermediate_schema),
        )
    }

    pub(crate) async fn multi_partitioned_join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        join_type: &JoinType,
        join_filter: Option<JoinFilter>,
        context: Arc<TaskContext>,
    ) -> Result<(Vec<String>, Vec<RecordBatch>, MetricsSet)> {
        let partition_count = 4;

        // Redistributing right input
        let right = Arc::new(RepartitionExec::try_new(
            right,
            Partitioning::RoundRobinBatch(partition_count),
        )?) as Arc<dyn ExecutionPlan>;

        // Use the required distribution for nested loop join to test partition data
        let nested_loop_join =
            NestedLoopJoinExec::try_new(left, right, join_filter, join_type, None)?;
        let columns = columns(&nested_loop_join.schema());
        let mut batches = vec![];
        for i in 0..partition_count {
            let stream = nested_loop_join.execute(i, Arc::clone(&context))?;
            let more_batches = common::collect(stream).await?;
            batches.extend(
                more_batches
                    .into_iter()
                    .inspect(|b| {
                        assert!(b.num_rows() <= context.session_config().batch_size())
                    })
                    .filter(|b| b.num_rows() > 0)
                    .collect::<Vec<_>>(),
            );
        }

        let metrics = nested_loop_join.metrics().unwrap();

        Ok((columns, batches, metrics))
    }

    fn new_task_ctx(batch_size: usize) -> Arc<TaskContext> {
        let base = TaskContext::default();
        // limit max size of intermediate batch used in nlj to 1
        let cfg = base.session_config().clone().with_batch_size(batch_size);
        Arc::new(base.with_session_config(cfg))
    }

    #[rstest]
    #[tokio::test]
    async fn join_inner_with_filter(#[values(1, 2, 16)] batch_size: usize) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        dbg!(&batch_size);
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::Inner,
            Some(filter),
            task_ctx,
        )
        .await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+----+----+----+
        | a1 | b1 | c1 | a2 | b2 | c2 |
        +----+----+----+----+----+----+
        | 5  | 5  | 50 | 2  | 2  | 80 |
        +----+----+----+----+----+----+
        "));

        assert_join_metrics!(metrics, 1);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_left_with_filter(#[values(1, 2, 16)] batch_size: usize) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::Left,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+----+----+----+
        | a1 | b1 | c1  | a2 | b2 | c2 |
        +----+----+-----+----+----+----+
        | 11 | 8  | 110 |    |    |    |
        | 5  | 5  | 50  | 2  | 2  | 80 |
        | 9  | 8  | 90  |    |    |    |
        +----+----+-----+----+----+----+
        "));

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_right_with_filter(#[values(1, 2, 16)] batch_size: usize) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::Right,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+----+----+-----+
        | a1 | b1 | c1 | a2 | b2 | c2  |
        +----+----+----+----+----+-----+
        |    |    |    | 10 | 10 | 100 |
        |    |    |    | 12 | 10 | 40  |
        | 5  | 5  | 50 | 2  | 2  | 80  |
        +----+----+----+----+----+-----+
        "));

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_full_with_filter(#[values(1, 2, 16)] batch_size: usize) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::Full,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+----+----+-----+
        | a1 | b1 | c1  | a2 | b2 | c2  |
        +----+----+-----+----+----+-----+
        |    |    |     | 10 | 10 | 100 |
        |    |    |     | 12 | 10 | 40  |
        | 11 | 8  | 110 |    |    |     |
        | 5  | 5  | 50  | 2  | 2  | 80  |
        | 9  | 8  | 90  |    |    |     |
        +----+----+-----+----+----+-----+
        "));

        assert_join_metrics!(metrics, 5);

        Ok(())
    }

    // Full join where the BUILD (left) side is split across multiple record
    // batches, so the build side is collected without concatenation. Exercises
    // resolving flat left-row indices to `(batch, row)` for matched rows and
    // slicing unmatched-left ranges within a single batch across batch boundaries.
    #[rstest]
    #[tokio::test]
    async fn join_full_with_filter_multi_batch_left(
        #[values(1, 2, 16)] batch_size: usize,
    ) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        // Split the build (left) side into single-row batches.
        let left = build_table(
            ("a1", &vec![5, 9, 11]),
            ("b1", &vec![5, 8, 8]),
            ("c1", &vec![50, 90, 110]),
            Some(1),
            Vec::new(),
        );
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::Full,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r#"
            +----+----+-----+----+----+-----+
            | a1 | b1 | c1  | a2 | b2 | c2  |
            +----+----+-----+----+----+-----+
            |    |    |     | 10 | 10 | 100 |
            |    |    |     | 12 | 10 | 40  |
            | 11 | 8  | 110 |    |    |     |
            | 5  | 5  | 50  | 2  | 2  | 80  |
            | 9  | 8  | 90  |    |    |     |
            +----+----+-----+----+----+-----+
            "#));

        assert_join_metrics!(metrics, 5);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_left_semi_with_filter(
        #[values(1, 2, 16)] batch_size: usize,
    ) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::LeftSemi,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+
        | a1 | b1 | c1 |
        +----+----+----+
        | 5  | 5  | 50 |
        +----+----+----+
        "));

        assert_join_metrics!(metrics, 1);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_left_anti_with_filter(
        #[values(1, 2, 16)] batch_size: usize,
    ) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::LeftAnti,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+
        | a1 | b1 | c1  |
        +----+----+-----+
        | 11 | 8  | 110 |
        | 9  | 8  | 90  |
        +----+----+-----+
        "));

        assert_join_metrics!(metrics, 2);

        Ok(())
    }

    #[tokio::test]
    async fn join_has_correct_stats() -> Result<()> {
        let left = build_left_table();
        let right = build_right_table();
        let nested_loop_join = NestedLoopJoinExec::try_new(
            left,
            right,
            None,
            &JoinType::Left,
            Some(vec![1, 2]),
        )?;
        let stats = nested_loop_join.partition_statistics(None)?;
        assert_eq!(
            nested_loop_join.schema().fields().len(),
            stats.column_statistics.len(),
        );
        assert_eq!(2, stats.column_statistics.len());
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_right_semi_with_filter(
        #[values(1, 2, 16)] batch_size: usize,
    ) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::RightSemi,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a2", "b2", "c2"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+
        | a2 | b2 | c2 |
        +----+----+----+
        | 2  | 2  | 80 |
        +----+----+----+
        "));

        assert_join_metrics!(metrics, 1);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_right_anti_with_filter(
        #[values(1, 2, 16)] batch_size: usize,
    ) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::RightAnti,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a2", "b2", "c2"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+
        | a2 | b2 | c2  |
        +----+----+-----+
        | 10 | 10 | 100 |
        | 12 | 10 | 40  |
        +----+----+-----+
        "));

        assert_join_metrics!(metrics, 2);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_left_mark_with_filter(
        #[values(1, 2, 16)] batch_size: usize,
    ) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::LeftMark,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a1", "b1", "c1", "mark"]);
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+-------+
        | a1 | b1 | c1  | mark  |
        +----+----+-----+-------+
        | 11 | 8  | 110 | false |
        | 5  | 5  | 50  | true  |
        | 9  | 8  | 90  | false |
        +----+----+-----+-------+
        "));

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    #[rstest]
    #[tokio::test]
    async fn join_right_mark_with_filter(
        #[values(1, 2, 16)] batch_size: usize,
    ) -> Result<()> {
        let task_ctx = new_task_ctx(batch_size);
        let left = build_left_table();
        let right = build_right_table();

        let filter = prepare_join_filter();
        let (columns, batches, metrics) = multi_partitioned_join_collect(
            left,
            right,
            &JoinType::RightMark,
            Some(filter),
            task_ctx,
        )
        .await?;
        assert_eq!(columns, vec!["a2", "b2", "c2", "mark"]);

        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+-------+
        | a2 | b2 | c2  | mark  |
        +----+----+-----+-------+
        | 10 | 10 | 100 | false |
        | 12 | 10 | 40  | false |
        | 2  | 2  | 80  | true  |
        +----+----+-----+-------+
        "));

        assert_join_metrics!(metrics, 3);

        Ok(())
    }

    #[tokio::test]
    async fn test_overallocation() -> Result<()> {
        let left = build_table(
            ("a1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            ("b1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            ("c1", &vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 0]),
            None,
            Vec::new(),
        );
        let right = build_table(
            ("a2", &vec![10, 11]),
            ("b2", &vec![12, 13]),
            ("c2", &vec![14, 15]),
            None,
            Vec::new(),
        );
        let filter = prepare_join_filter();

        // Join types that support memory-limited fallback should succeed
        // even under tight memory limits (they spill to disk instead of OOM).
        let fallback_join_types = vec![
            JoinType::Inner,
            JoinType::Left,
            JoinType::LeftSemi,
            JoinType::LeftAnti,
            JoinType::LeftMark,
            JoinType::Right,
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::RightMark,
        ];

        for join_type in &fallback_join_types {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(100, 1.0)
                .build_arc()?;
            let task_ctx = TaskContext::default().with_runtime(runtime);
            let task_ctx = Arc::new(task_ctx);

            // Should succeed via spill fallback, not OOM
            let _result = multi_partitioned_join_collect(
                Arc::clone(&left),
                Arc::clone(&right),
                join_type,
                Some(filter.clone()),
                task_ctx,
            )
            .await?;
        }

        // FULL JOIN with multiple right partitions is intentionally not
        // supported in the fallback path yet (cross-partition left-bitmap
        // coordination is missing). It should still OOM under tight memory.
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(100, 1.0)
            .build_arc()?;
        let task_ctx = TaskContext::default().with_runtime(runtime);
        let task_ctx = Arc::new(task_ctx);
        let err = multi_partitioned_join_collect(
            Arc::clone(&left),
            Arc::clone(&right),
            &JoinType::Full,
            Some(filter.clone()),
            task_ctx,
        )
        .await
        .unwrap_err();
        assert_contains!(err.to_string(), "Resources exhausted");

        Ok(())
    }

    /// Returns the column names on the schema
    fn columns(schema: &Schema) -> Vec<String> {
        schema.fields().iter().map(|f| f.name().clone()).collect()
    }

    // ========================================================================
    // Memory-limited execution tests
    // ========================================================================

    /// Helper to run a NLJ using partition 0 and collect results + metrics.
    async fn join_collect(
        left: Arc<dyn ExecutionPlan>,
        right: Arc<dyn ExecutionPlan>,
        join_type: &JoinType,
        join_filter: Option<JoinFilter>,
        context: Arc<TaskContext>,
    ) -> Result<(Vec<String>, Vec<RecordBatch>, MetricsSet)> {
        let nested_loop_join =
            NestedLoopJoinExec::try_new(left, right, join_filter, join_type, None)?;
        let columns = columns(&nested_loop_join.schema());
        let stream = nested_loop_join.execute(0, context)?;
        let batches: Vec<RecordBatch> = common::collect(stream)
            .await?
            .into_iter()
            .filter(|b| b.num_rows() > 0)
            .collect();
        let metrics = nested_loop_join.metrics().unwrap();
        Ok((columns, batches, metrics))
    }

    /// Create a TaskContext with tight memory limit and disk spilling enabled.
    fn task_ctx_with_memory_limit(
        memory_limit: usize,
        batch_size: usize,
    ) -> Result<Arc<TaskContext>> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(memory_limit, 1.0)
            .build_arc()?;
        let cfg = TaskContext::default()
            .session_config()
            .clone()
            .with_batch_size(batch_size);
        let task_ctx = TaskContext::default()
            .with_runtime(runtime)
            .with_session_config(cfg);
        Ok(Arc::new(task_ctx))
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_inner_join() -> Result<()> {
        // Use a very small memory limit to force OOM → fallback to spill.
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::Inner, Some(filter), task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Verify spill actually occurred (memory-limited path was taken)
        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "Expected spilling to occur under tight memory limit"
        );

        // Result should be identical to the non-memory-limited case
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+----+----+----+
        | a1 | b1 | c1 | a2 | b2 | c2 |
        +----+----+----+----+----+----+
        | 5  | 5  | 50 | 2  | 2  | 80 |
        +----+----+----+----+----+----+
        "));
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_left_join() -> Result<()> {
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::Left, Some(filter), task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Verify spill actually occurred
        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "Expected spilling to occur under tight memory limit"
        );

        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+----+----+----+
        | a1 | b1 | c1  | a2 | b2 | c2 |
        +----+----+-----+----+----+----+
        | 11 | 8  | 110 |    |    |    |
        | 5  | 5  | 50  | 2  | 2  | 80 |
        | 9  | 8  | 90  |    |    |    |
        +----+----+-----+----+----+----+
        "));
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_fits_in_memory_no_spill() -> Result<()> {
        // Use a large memory limit — everything fits, no spilling needed.
        let task_ctx = task_ctx_with_memory_limit(10_000_000, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::Inner, Some(filter), task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Verify no spilling occurred (standard OnceFut path was used)
        assert_eq!(
            metrics.spill_count().unwrap_or(0),
            0,
            "Expected no spilling with generous memory limit"
        );

        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+----+----+----+
        | a1 | b1 | c1 | a2 | b2 | c2 |
        +----+----+----+----+----+----+
        | 5  | 5  | 50 | 2  | 2  | 80 |
        +----+----+----+----+----+----+
        "));
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_empty_inputs() -> Result<()> {
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;

        // Empty left table
        let empty_left = build_table(
            ("a1", &vec![]),
            ("b1", &vec![]),
            ("c1", &vec![]),
            None,
            Vec::new(),
        );
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (_columns, batches, _metrics) =
            join_collect(empty_left, right, &JoinType::Inner, Some(filter), task_ctx)
                .await?;
        assert!(batches.is_empty() || batches.iter().all(|b| b.num_rows() == 0));

        // Empty right table
        let task_ctx2 = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let empty_right = build_table(
            ("a2", &vec![]),
            ("b2", &vec![]),
            ("c2", &vec![]),
            None,
            Vec::new(),
        );
        let filter2 = prepare_join_filter();

        let (_columns, batches, _metrics) = join_collect(
            left,
            empty_right,
            &JoinType::Inner,
            Some(filter2),
            task_ctx2,
        )
        .await?;
        assert!(batches.is_empty() || batches.iter().all(|b| b.num_rows() == 0));

        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_no_disk_falls_back_to_oom() -> Result<()> {
        // When disk is disabled, fallback is not possible and OOM should occur.
        use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};

        let runtime = RuntimeEnvBuilder::new()
            .with_memory_limit(100, 1.0)
            .with_disk_manager_builder(
                DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
            )
            .build_arc()?;
        let task_ctx = Arc::new(TaskContext::default().with_runtime(runtime));

        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let err = join_collect(left, right, &JoinType::Inner, Some(filter), task_ctx)
            .await
            .unwrap_err();

        assert_contains!(err.to_string(), "Resources exhausted");
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_right_join() -> Result<()> {
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::Right, Some(filter), task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Verify spill actually occurred
        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "Expected spilling to occur under tight memory limit"
        );

        // Right join: all right rows appear. Unmatched right rows get NULLs on left.
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+----+----+-----+
        | a1 | b1 | c1 | a2 | b2 | c2  |
        +----+----+----+----+----+-----+
        |    |    |    | 10 | 10 | 100 |
        |    |    |    | 12 | 10 | 40  |
        | 5  | 5  | 50 | 2  | 2  | 80  |
        +----+----+----+----+----+-----+
        "));
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_full_join() -> Result<()> {
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::Full, Some(filter), task_ctx).await?;

        assert_eq!(columns, vec!["a1", "b1", "c1", "a2", "b2", "c2"]);

        // Verify spill actually occurred
        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "Expected spilling to occur under tight memory limit"
        );

        // Full join: unmatched from both sides appear with NULL padding.
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+----+----+-----+
        | a1 | b1 | c1  | a2 | b2 | c2  |
        +----+----+-----+----+----+-----+
        |    |    |     | 10 | 10 | 100 |
        |    |    |     | 12 | 10 | 40  |
        | 11 | 8  | 110 |    |    |     |
        | 5  | 5  | 50  | 2  | 2  | 80  |
        | 9  | 8  | 90  |    |    |     |
        +----+----+-----+----+----+-----+
        "));
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_right_semi_join() -> Result<()> {
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::RightSemi, Some(filter), task_ctx)
                .await?;

        assert_eq!(columns, vec!["a2", "b2", "c2"]);

        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "Expected spilling to occur under tight memory limit"
        );

        // Right semi: only right rows that matched at least one left row.
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+----+
        | a2 | b2 | c2 |
        +----+----+----+
        | 2  | 2  | 80 |
        +----+----+----+
        "));
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_right_anti_join() -> Result<()> {
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::RightAnti, Some(filter), task_ctx)
                .await?;

        assert_eq!(columns, vec!["a2", "b2", "c2"]);

        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "Expected spilling to occur under tight memory limit"
        );

        // Right anti: right rows that did NOT match any left row.
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+
        | a2 | b2 | c2  |
        +----+----+-----+
        | 10 | 10 | 100 |
        | 12 | 10 | 40  |
        +----+----+-----+
        "));
        Ok(())
    }

    #[tokio::test]
    async fn test_nlj_memory_limited_right_mark_join() -> Result<()> {
        let task_ctx = task_ctx_with_memory_limit(50, 16)?;
        let left = build_left_table();
        let right = build_right_table();
        let filter = prepare_join_filter();

        let (columns, batches, metrics) =
            join_collect(left, right, &JoinType::RightMark, Some(filter), task_ctx)
                .await?;

        assert_eq!(columns, vec!["a2", "b2", "c2", "mark"]);

        assert!(
            metrics.spill_count().unwrap_or(0) > 0,
            "Expected spilling to occur under tight memory limit"
        );

        // Right mark: all right rows with a bool column indicating match.
        allow_duplicates!(assert_snapshot!(batches_to_sort_string(&batches), @r"
        +----+----+-----+-------+
        | a2 | b2 | c2  | mark  |
        +----+----+-----+-------+
        | 10 | 10 | 100 | false |
        | 12 | 10 | 40  | false |
        | 2  | 2  | 80  | true  |
        +----+----+-----+-------+
        "));
        Ok(())
    }

    /// Tests for chunking the broadcast of large variable-length build values,
    /// which would otherwise overflow `i32` offsets ("offset overflow" panic in
    /// `OffsetBuffer::from_repeated_length`).
    mod broadcast_chunking {
        use super::*;
        use arrow::array::{BinaryArray, Int32Array};
        use datafusion_common::cast::as_int32_array;
        use datafusion_physical_expr::expressions::IsNotNullExpr;

        /// Build (left) side with one row holding a 10-byte binary payload,
        /// probe (right) side with 10 rows, and the joined output layout
        fn broadcast_test_inputs() -> (RecordBatch, RecordBatch, Schema, Vec<ColumnIndex>)
        {
            let left_schema = Arc::new(Schema::new(vec![
                Field::new("a1", DataType::Int32, true),
                Field::new("blob", DataType::Binary, true),
            ]));
            let left_batch = RecordBatch::try_new(
                left_schema,
                vec![
                    Arc::new(Int32Array::from(vec![5])),
                    Arc::new(BinaryArray::from_opt_vec(vec![Some(b"0123456789")])),
                ],
            )
            .unwrap();

            let right_schema =
                Arc::new(Schema::new(vec![Field::new("b2", DataType::Int32, true)]));
            let right_batch = RecordBatch::try_new(
                right_schema,
                vec![Arc::new(Int32Array::from((0..10).collect::<Vec<_>>()))],
            )
            .unwrap();

            let output_schema = Schema::new(vec![
                Field::new("a1", DataType::Int32, true),
                Field::new("blob", DataType::Binary, true),
                Field::new("b2", DataType::Int32, true),
            ]);
            let col_indices = vec![
                ColumnIndex {
                    index: 0,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 0,
                    side: JoinSide::Right,
                },
            ];
            (left_batch, right_batch, output_schema, col_indices)
        }

        #[test]
        fn chunks_large_broadcast_values() -> Result<()> {
            let (left, right, output_schema, col_indices) = broadcast_test_inputs();

            // The build value is 10 bytes, so a 32-byte limit allows 3 rows
            // per chunk
            let chunked = build_row_join_batch_with_limit(
                &output_schema,
                &left,
                0,
                &right,
                None,
                &col_indices,
                JoinSide::Left,
                32,
            )?;
            assert_eq!(
                chunked.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
                vec![3, 3, 3, 1]
            );

            // The chunked result must concatenate back to the unchunked result
            let unchunked = build_row_join_batch_with_limit(
                &output_schema,
                &left,
                0,
                &right,
                None,
                &col_indices,
                JoinSide::Left,
                usize::MAX,
            )?;
            assert_eq!(unchunked.len(), 1);
            let concatenated = concat_batches(&Arc::new(output_schema), &chunked)?;
            assert_eq!(concatenated, unchunked[0]);
            Ok(())
        }

        #[test]
        fn chunks_after_probe_filter() -> Result<()> {
            let (left, right, output_schema, col_indices) = broadcast_test_inputs();

            // Keep only even probe rows: 0, 2, 4, 6, 8
            let mask =
                BooleanArray::from((0..10).map(|i| i % 2 == 0).collect::<Vec<_>>());
            let chunked = build_row_join_batch_with_limit(
                &output_schema,
                &left,
                0,
                &right,
                Some(mask.clone()),
                &col_indices,
                JoinSide::Left,
                32,
            )?;
            assert_eq!(
                chunked.iter().map(|b| b.num_rows()).collect::<Vec<_>>(),
                vec![3, 2]
            );

            let concatenated =
                concat_batches(&Arc::new(output_schema.clone()), &chunked)?;
            let unchunked = build_row_join_batch_with_limit(
                &output_schema,
                &left,
                0,
                &right,
                Some(mask),
                &col_indices,
                JoinSide::Left,
                usize::MAX,
            )?;
            assert_eq!(concatenated, unchunked[0]);
            assert_eq!(
                as_int32_array(concatenated.column(2))?.values(),
                &[0, 2, 4, 6, 8]
            );
            Ok(())
        }

        #[test]
        fn broadcast_edge_cases() -> Result<()> {
            let (left, right, output_schema, col_indices) = broadcast_test_inputs();

            // A null build value contributes no bytes: never chunks, even with
            // a 1-byte limit
            let null_left = RecordBatch::try_new(
                left.schema(),
                vec![
                    Arc::new(Int32Array::from(vec![5])),
                    Arc::new(BinaryArray::from_opt_vec(vec![None])),
                ],
            )?;
            let batches = build_row_join_batch_with_limit(
                &output_schema,
                &null_left,
                0,
                &right,
                None,
                &col_indices,
                JoinSide::Left,
                1,
            )?;
            assert_eq!(batches.len(), 1);
            assert_eq!(batches[0].num_rows(), 10);

            // A value longer than the limit degrades to one row per chunk
            let batches = build_row_join_batch_with_limit(
                &output_schema,
                &left,
                0,
                &right,
                None,
                &col_indices,
                JoinSide::Left,
                5,
            )?;
            assert_eq!(batches.len(), 10);
            assert!(batches.iter().all(|b| b.num_rows() == 1));

            // An all-filtered probe batch produces no output
            let batches = build_row_join_batch_with_limit(
                &output_schema,
                &left,
                0,
                &right,
                Some(BooleanArray::from(vec![false; 10])),
                &col_indices,
                JoinSide::Left,
                32,
            )?;
            assert!(batches.is_empty());
            Ok(())
        }

        #[test]
        fn chunked_filter_mask_matches_unchunked() -> Result<()> {
            let (left, _, _, _) = broadcast_test_inputs();

            // Right batch with a null to exercise null handling of the filter
            // result across chunk boundaries
            let right_schema =
                Arc::new(Schema::new(vec![Field::new("b2", DataType::Int32, true)]));
            let right = RecordBatch::try_new(
                right_schema,
                vec![Arc::new(Int32Array::from(
                    (0..10)
                        .map(|i| if i == 5 { None } else { Some(i) })
                        .collect::<Vec<_>>(),
                ))],
            )?;

            // Filter references the left blob so the intermediate batch has to
            // broadcast it, and evaluates `b2 >= 4` on the right column
            let filter_schema = Schema::new(vec![
                Field::new("blob", DataType::Binary, true),
                Field::new("b2", DataType::Int32, true),
            ]);
            let column_indices = vec![
                ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 0,
                    side: JoinSide::Right,
                },
            ];
            let expr = Arc::new(BinaryExpr::new(
                Arc::new(Column::new("b2", 1)),
                Operator::GtEq,
                Arc::new(Literal::new(ScalarValue::Int32(Some(4)))),
            )) as Arc<dyn PhysicalExpr>;
            let filter = JoinFilter::new(expr, column_indices, Arc::new(filter_schema));

            let chunked =
                apply_filter_to_row_join_batch_with_limit(&left, 0, &right, &filter, 32)?;
            let unchunked = apply_filter_to_row_join_batch_with_limit(
                &left,
                0,
                &right,
                &filter,
                usize::MAX,
            )?;
            assert_eq!(chunked, unchunked);

            // `b2 >= 4` is null for the null row, which must combine to false
            let expected =
                BooleanArray::from((0..10).map(|i| i >= 4 && i != 5).collect::<Vec<_>>());
            assert_eq!(chunked, expected);
            Ok(())
        }

        #[test]
        fn i32_offset_type_detection() {
            use arrow::datatypes::Fields;

            assert!(contains_i32_offset_data(&DataType::Utf8));
            assert!(contains_i32_offset_data(&DataType::Binary));
            assert!(contains_i32_offset_data(&DataType::List(Arc::new(
                Field::new_list_field(DataType::Int32, true)
            ))));
            assert!(contains_i32_offset_data(&DataType::LargeList(Arc::new(
                Field::new_list_field(DataType::Utf8, true)
            ))));
            assert!(contains_i32_offset_data(&DataType::Struct(Fields::from(
                vec![Field::new("a", DataType::Binary, true)]
            ))));

            assert!(!contains_i32_offset_data(&DataType::Int32));
            assert!(!contains_i32_offset_data(&DataType::LargeUtf8));
            assert!(!contains_i32_offset_data(&DataType::Utf8View));
            assert!(!contains_i32_offset_data(&DataType::LargeList(Arc::new(
                Field::new_list_field(DataType::Int64, true)
            ))));
        }

        /// Reproducer for the "offset overflow" panic: broadcasting a single
        /// large `Binary` value across a whole probe batch used to overflow
        /// the `i32` offsets both of the filter's intermediate batch and of
        /// the output batch.
        ///
        /// Ignored by default: it materializes ~2.5 GiB of join output
        /// (several GiB peak RSS). Run with:
        /// `cargo test -p datafusion-physical-plan --release -- --ignored nlj_broadcast_offset_overflow`
        #[tokio::test]
        #[ignore = "allocates several GiB of memory"]
        async fn nlj_broadcast_offset_overflow() -> Result<()> {
            let value = vec![42u8; 300 * 1024];
            let left_schema = Arc::new(Schema::new(vec![Field::new(
                "blob",
                DataType::Binary,
                true,
            )]));
            let left_batch = RecordBatch::try_new(
                Arc::clone(&left_schema),
                vec![Arc::new(BinaryArray::from_opt_vec(vec![Some(&value)]))],
            )?;
            let left: Arc<dyn ExecutionPlan> =
                TestMemoryExec::try_new_exec(&[vec![left_batch]], left_schema, None)?;

            // A single 8192-row probe batch: 300 KiB * 8192 > i32::MAX bytes
            let n_right = 8192;
            let right_schema =
                Arc::new(Schema::new(vec![Field::new("b2", DataType::Int32, true)]));
            let right_batch = RecordBatch::try_new(
                Arc::clone(&right_schema),
                vec![Arc::new(Int32Array::from(
                    (0..n_right as i32).collect::<Vec<_>>(),
                ))],
            )?;
            let right: Arc<dyn ExecutionPlan> =
                TestMemoryExec::try_new_exec(&[vec![right_batch]], right_schema, None)?;

            // The filter references the blob column so the filter's
            // intermediate batch also broadcasts the large value
            let filter_schema =
                Schema::new(vec![Field::new("blob", DataType::Binary, true)]);
            let filter_expr =
                Arc::new(IsNotNullExpr::new(Arc::new(Column::new("blob", 0))))
                    as Arc<dyn PhysicalExpr>;
            let filter = JoinFilter::new(
                filter_expr,
                vec![ColumnIndex {
                    index: 0,
                    side: JoinSide::Left,
                }],
                Arc::new(filter_schema),
            );

            let join = NestedLoopJoinExec::try_new(
                left,
                right,
                Some(filter),
                &JoinType::Inner,
                None,
            )?;
            let ctx = Arc::new(TaskContext::default());
            let batch_size = ctx.session_config().batch_size();
            let stream = join.execute(0, ctx)?;
            let batches = common::collect(stream).await?;

            let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
            assert_eq!(total_rows, n_right);
            for batch in &batches {
                assert!(batch.num_rows() <= batch_size);
                let blob = batch.column(0).as_binary::<i32>();
                // Each output batch stays under the i32 offset limit
                assert!(blob.value_data().len() <= i32::MAX as usize);
                assert_eq!(blob.value(0).len(), value.len());
            }
            Ok(())
        }
    }
}
