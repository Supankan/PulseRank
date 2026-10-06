"""Initial database schema for PulseRank

Revision ID: 001_initial
Revises: 
Create Date: 2026-09-10 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users
    op.create_table(
        'users',
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('history_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('user_id')
    )

    # Articles
    op.create_table(
        'articles',
        sa.Column('article_id', sa.String(length=64), nullable=False),
        sa.Column('category', sa.String(length=64), nullable=False),
        sa.Column('subcategory', sa.String(length=64), nullable=False),
        sa.Column('title', sa.Text(), nullable=False),
        sa.Column('abstract', sa.Text(), nullable=False),
        sa.Column('url', sa.String(length=512), nullable=True),
        sa.Column('title_entities', sa.JSON(), nullable=True),
        sa.Column('abstract_entities', sa.JSON(), nullable=True),
        sa.Column('published_at', sa.DateTime(), nullable=True),
        sa.Column('click_count', sa.Integer(), nullable=False, server_default='0'),
        sa.PrimaryKeyConstraint('article_id')
    )
    op.create_index(op.f('ix_articles_category'), 'articles', ['category'], unique=False)
    op.create_index(op.f('ix_articles_subcategory'), 'articles', ['subcategory'], unique=False)

    # Interactions
    op.create_table(
        'interactions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('article_id', sa.String(length=64), nullable=False),
        sa.Column('interaction_type', sa.String(length=32), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['article_id'], ['articles.article_id']),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_interactions_user_id'), 'interactions', ['user_id'], unique=False)
    op.create_index(op.f('ix_interactions_article_id'), 'interactions', ['article_id'], unique=False)
    op.create_index(op.f('ix_interactions_interaction_type'), 'interactions', ['interaction_type'], unique=False)
    op.create_index(op.f('ix_interactions_timestamp'), 'interactions', ['timestamp'], unique=False)

    # Recommendation Runs
    op.create_table(
        'recommendation_runs',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('user_id', sa.String(length=64), nullable=False),
        sa.Column('model_name', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_recommendation_runs_user_id'), 'recommendation_runs', ['user_id'], unique=False)
    op.create_index(op.f('ix_recommendation_runs_model_name'), 'recommendation_runs', ['model_name'], unique=False)

    # Recommendation Outputs
    op.create_table(
        'recommendation_outputs',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('recommendation_run_id', sa.String(length=64), nullable=False),
        sa.Column('article_id', sa.String(length=64), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('score', sa.Float(), nullable=False),
        sa.Column('sources', sa.JSON(), nullable=False),
        sa.Column('explanation', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['article_id'], ['articles.article_id']),
        sa.ForeignKeyConstraint(['recommendation_run_id'], ['recommendation_runs.id']),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_recommendation_outputs_recommendation_run_id'), 'recommendation_outputs', ['recommendation_run_id'], unique=False)

    # Model Metadata
    op.create_table(
        'model_metadata',
        sa.Column('id', sa.String(length=64), nullable=False),
        sa.Column('name', sa.String(length=64), nullable=False),
        sa.Column('model_type', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('artifact_path', sa.String(length=256), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )

    # Evaluation Results
    op.create_table(
        'evaluation_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('model_name', sa.String(length=64), nullable=False),
        sa.Column('auc', sa.Float(), nullable=False),
        sa.Column('mrr', sa.Float(), nullable=False),
        sa.Column('ndcg_at_5', sa.Float(), nullable=False),
        sa.Column('ndcg_at_10', sa.Float(), nullable=False),
        sa.Column('recall_at_k', sa.Float(), nullable=False),
        sa.Column('coverage', sa.Float(), nullable=False),
        sa.Column('num_impressions', sa.Integer(), nullable=False),
        sa.Column('num_users', sa.Integer(), nullable=False),
        sa.Column('evaluated_at', sa.DateTime(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_evaluation_results_model_name'), 'evaluation_results', ['model_name'], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    existing_tables = set(inspector.get_table_names())
    for tbl in [
        'evaluation_results',
        'model_metadata',
        'recommendation_outputs',
        'recommendation_runs',
        'interactions',
        'articles',
        'users',
    ]:
        if tbl in existing_tables:
            op.drop_table(tbl)
