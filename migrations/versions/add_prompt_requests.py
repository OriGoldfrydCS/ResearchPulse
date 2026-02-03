"""Add prompt_requests table for system controller compliance

Revision ID: add_prompt_requests
Revises: add_profile_fields
Create Date: 2026-02-02

This migration adds the prompt_requests table to store parsed research prompts
for audit trail, template analytics, and output enforcement compliance tracking.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_prompt_requests'
down_revision = 'add_profile_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create prompt_requests table."""
    op.create_table(
        'prompt_requests',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('run_id', sa.String(100), nullable=True),
        
        # Raw prompt
        sa.Column('raw_prompt', sa.Text(), nullable=False),
        
        # Parsed template classification
        sa.Column('template', sa.String(50), nullable=False),
        
        # Extracted parameters
        sa.Column('topic', sa.Text(), nullable=True),
        sa.Column('venue', sa.String(255), nullable=True),
        sa.Column('time_period', sa.String(100), nullable=True),
        sa.Column('time_days', sa.Integer(), nullable=True),
        sa.Column('requested_count', sa.Integer(), nullable=True),
        sa.Column('output_count', sa.Integer(), nullable=True),
        sa.Column('retrieval_count', sa.Integer(), nullable=True),
        sa.Column('method_or_approach', sa.Text(), nullable=True),
        sa.Column('application_domain', sa.Text(), nullable=True),
        
        # Request type flags
        sa.Column('is_survey_request', sa.Boolean(), server_default='false'),
        sa.Column('is_trends_request', sa.Boolean(), server_default='false'),
        sa.Column('is_structured_output', sa.Boolean(), server_default='false'),
        
        # Compliance tracking
        sa.Column('output_enforced', sa.Boolean(), server_default='false'),
        sa.Column('output_insufficient', sa.Boolean(), server_default='false'),
        sa.Column('compliance_status', sa.String(50), server_default='pending'),
        sa.Column('compliance_message', sa.Text(), nullable=True),
        
        # Actual results
        sa.Column('papers_retrieved', sa.Integer(), nullable=True),
        sa.Column('papers_returned', sa.Integer(), nullable=True),
        
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
        
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
    )
    
    # Create indexes
    op.create_index('ix_prompt_request_user_id', 'prompt_requests', ['user_id'])
    op.create_index('ix_prompt_request_run_id', 'prompt_requests', ['run_id'])
    op.create_index('ix_prompt_request_template', 'prompt_requests', ['template'])
    op.create_index('ix_prompt_request_created_at', 'prompt_requests', ['created_at'])
    op.create_index('ix_prompt_request_compliance', 'prompt_requests', ['compliance_status'])


def downgrade() -> None:
    """Drop prompt_requests table."""
    op.drop_index('ix_prompt_request_compliance', table_name='prompt_requests')
    op.drop_index('ix_prompt_request_created_at', table_name='prompt_requests')
    op.drop_index('ix_prompt_request_template', table_name='prompt_requests')
    op.drop_index('ix_prompt_request_run_id', table_name='prompt_requests')
    op.drop_index('ix_prompt_request_user_id', table_name='prompt_requests')
    op.drop_table('prompt_requests')
