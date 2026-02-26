"""Add autonomous components tables.

This migration adds tables for the four autonomous components:
1. run_audit_logs - Structured run summaries
2. profile_evolution_suggestions - Advisory profile change suggestions
3. live_documents - Living research document
4. live_document_history - Version history for live documents

Also extends paper_views with LLM novelty scoring columns.

Revision ID: add_autonomous_components
Revises: paper_feedback_history
Create Date: 2026-02-12

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_autonomous_components'
down_revision = 'add_paper_summary'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ==========================================================================
    # Table: run_audit_logs
    # ==========================================================================
    op.create_table(
        'run_audit_logs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('run_id', sa.String(100), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        
        # Summary statistics
        sa.Column('papers_retrieved_count', sa.Integer(), server_default='0'),
        sa.Column('papers_scored_count', sa.Integer(), server_default='0'),
        sa.Column('papers_shared_count', sa.Integer(), server_default='0'),
        sa.Column('papers_discarded_count', sa.Integer(), server_default='0'),
        
        # Detailed JSON data
        sa.Column('papers_retrieved', postgresql.JSONB(), server_default='[]'),
        sa.Column('papers_shared', postgresql.JSONB(), server_default='[]'),
        sa.Column('papers_discarded', postgresql.JSONB(), server_default='[]'),
        sa.Column('colleague_shares', postgresql.JSONB(), server_default='{}'),
        
        # Execution metadata
        sa.Column('execution_time_ms', sa.Integer()),
        sa.Column('llm_calls_count', sa.Integer(), server_default='0'),
        sa.Column('llm_tokens_used', sa.Integer(), server_default='0'),
        sa.Column('stop_reason', sa.Text()),
        
        # User profile snapshot (for audit purposes)
        sa.Column('user_profile_snapshot', postgresql.JSONB(), server_default='{}'),
        
        # Full structured log
        sa.Column('full_log', postgresql.JSONB(), nullable=False),
    )
    
    op.create_index('ix_run_audit_logs_user_id', 'run_audit_logs', ['user_id'])
    op.create_index('ix_run_audit_logs_run_id', 'run_audit_logs', ['run_id'])
    op.create_index('ix_run_audit_logs_created_at', 'run_audit_logs', ['created_at'])
    
    # ==========================================================================
    # Table: profile_evolution_suggestions
    # ==========================================================================
    op.create_table(
        'profile_evolution_suggestions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('run_id', sa.String(100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        
        # Suggestion content
        sa.Column('suggestion_type', sa.String(50), nullable=False),  # add_topic, remove_topic, etc.
        sa.Column('suggestion_text', sa.Text(), nullable=False),
        sa.Column('reasoning', sa.Text(), nullable=False),
        sa.Column('confidence', sa.Float()),
        
        # Supporting evidence
        sa.Column('supporting_papers', postgresql.JSONB(), server_default='[]'),
        
        # Structured suggestion data
        sa.Column('suggestion_data', postgresql.JSONB(), nullable=False),
        
        # Status tracking
        sa.Column('status', sa.String(50), server_default="'pending'"),  # pending, accepted, rejected, expired
        sa.Column('reviewed_at', sa.DateTime(timezone=True)),
        sa.Column('reviewed_by', sa.String(100)),  # For multi-user future
    )
    
    op.create_index('ix_profile_suggestions_user_id', 'profile_evolution_suggestions', ['user_id'])
    op.create_index('ix_profile_suggestions_status', 'profile_evolution_suggestions', ['status'])
    op.create_index('ix_profile_suggestions_created_at', 'profile_evolution_suggestions', ['created_at'])
    op.create_index('ix_profile_suggestions_user_status', 'profile_evolution_suggestions', ['user_id', 'status'])
    
    # ==========================================================================
    # Table: live_documents
    # ==========================================================================
    op.create_table(
        'live_documents',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True),
        
        # Document metadata
        sa.Column('title', sa.String(500), server_default="'ResearchPulse - Live Briefing'"),
        
        # Document content - JSONB for full data, text for rendered
        sa.Column('document_data', postgresql.JSONB(), server_default='{}'),
        sa.Column('markdown_content', sa.Text(), nullable=False),
        sa.Column('html_content', sa.Text()),
        
        # Timestamps
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    
    op.create_index('ix_live_documents_user_id', 'live_documents', ['user_id'])
    
    # ==========================================================================
    # Table: live_document_history
    # ==========================================================================
    op.create_table(
        'live_document_history',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('document_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('live_documents.id', ondelete='CASCADE'), nullable=False),
        
        # Stored content
        sa.Column('document_data', postgresql.JSONB(), server_default='{}'),
        sa.Column('markdown_content', sa.Text()),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()')),
    )
    
    op.create_index('ix_live_doc_history_doc_id', 'live_document_history', ['document_id'])
    
    # ==========================================================================
    # Extend paper_views with LLM novelty columns
    # ==========================================================================
    op.add_column('paper_views', sa.Column('llm_novelty_score', sa.Float(), nullable=True))
    op.add_column('paper_views', sa.Column('llm_novelty_reasoning', sa.Text(), nullable=True))
    op.add_column('paper_views', sa.Column('novelty_sub_scores', postgresql.JSONB(), server_default='{}'))


def downgrade() -> None:
    # Remove paper_views columns
    op.drop_column('paper_views', 'novelty_sub_scores')
    op.drop_column('paper_views', 'llm_novelty_reasoning')
    op.drop_column('paper_views', 'llm_novelty_score')
    
    # Drop tables in reverse order
    op.drop_table('live_document_history')
    op.drop_table('live_documents')
    op.drop_table('profile_evolution_suggestions')
    op.drop_table('run_audit_logs')
