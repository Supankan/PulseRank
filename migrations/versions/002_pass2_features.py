"""Pass 2 feature expansion schema migration

Revision ID: 002_pass2_features
Revises: 001_initial
Create Date: 2026-09-11 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '002_pass2_features'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update evaluation_results
    with op.batch_alter_table('evaluation_results') as batch_op:
        batch_op.add_column(sa.Column('dataset_source', sa.String(length=64), nullable=False, server_default='mind-small'))
        batch_op.add_column(sa.Column('evaluation_mode', sa.String(length=32), nullable=False, server_default='impression'))
        batch_op.add_column(sa.Column('impression_coverage', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('catalog_coverage', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('hit_rate_at_k', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('num_scored_impressions', sa.Integer(), nullable=True))

    # 2. Annotations
    op.create_table(
        'annotations',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('article_id', sa.String(length=64), nullable=False),
        sa.Column('task_type', sa.String(length=32), nullable=False),
        sa.Column('predicted_label', sa.String(length=64), nullable=True),
        sa.Column('true_label', sa.String(length=64), nullable=True),
        sa.Column('annotator_id', sa.String(length=64), nullable=False, server_default='human'),
        sa.Column('uncertainty_score', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('meta_info', sa.JSON(), nullable=True),
        sa.ForeignKeyConstraint(['article_id'], ['articles.article_id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_annotations_article_id'), 'annotations', ['article_id'], unique=False)
    op.create_index(op.f('ix_annotations_task_type'), 'annotations', ['task_type'], unique=False)

    # 3. Experiment Runs
    op.create_table(
        'experiment_runs',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('experiment_name', sa.String(length=64), nullable=False),
        sa.Column('variant_a', sa.String(length=64), nullable=False),
        sa.Column('variant_b', sa.String(length=64), nullable=False),
        sa.Column('traffic_split', sa.Float(), nullable=False, server_default='0.5'),
        sa.Column('metrics_summary', sa.JSON(), nullable=True),
        sa.Column('p_value', sa.Float(), nullable=True),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='running'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_experiment_runs_experiment_name'), 'experiment_runs', ['experiment_name'], unique=False)

    # 4. Model Registry
    op.create_table(
        'model_registry',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('model_name', sa.String(length=64), nullable=False),
        sa.Column('version', sa.String(length=32), nullable=False),
        sa.Column('stage', sa.String(length=32), nullable=False, server_default='staging'),
        sa.Column('artifact_path', sa.String(length=256), nullable=True),
        sa.Column('manifest_data', sa.JSON(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_model_registry_model_name'), 'model_registry', ['model_name'], unique=False)

    # 5. Monitoring Logs
    op.create_table(
        'monitoring_logs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('endpoint', sa.String(length=64), nullable=False),
        sa.Column('latency_ms', sa.Float(), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=True),
        sa.Column('channel_distribution', sa.JSON(), nullable=True),
        sa.Column('prediction_entropy', sa.Float(), nullable=True),
        sa.Column('drift_psi', sa.Float(), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_monitoring_logs_endpoint'), 'monitoring_logs', ['endpoint'], unique=False)
    op.create_index(op.f('ix_monitoring_logs_timestamp'), 'monitoring_logs', ['timestamp'], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())

    # Drop Pass-2 added tables if they exist
    for tbl in ['monitoring_logs', 'model_registry', 'experiment_runs', 'annotations']:
        if tbl in existing_tables:
            op.drop_table(tbl)

    # Drop Pass-2 added columns from evaluation_results if they exist
    if 'evaluation_results' in existing_tables:
        existing_cols = {c['name'] for c in inspector.get_columns('evaluation_results')}
        cols_to_drop = [
            'num_scored_impressions',
            'hit_rate_at_k',
            'catalog_coverage',
            'impression_coverage',
            'evaluation_mode',
            'dataset_source',
        ]
        to_drop = [col for col in cols_to_drop if col in existing_cols]
        if to_drop:
            with op.batch_alter_table('evaluation_results') as batch_op:
                for col in to_drop:
                    batch_op.drop_column(col)
