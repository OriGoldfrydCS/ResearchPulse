"""Add colleague_email_log table and interest_headline column.

Revision ID: colleague_email_log
Revises: add_colleague_onboarding
Create Date: 2026-02-10

This migration adds:
1. colleague_email_log table - tracks outbound emails to colleagues for activity reporting
2. interest_headline column on colleagues - caches LLM-generated headline for UI display

State Model Notes:
- Categories are DERIVED from interests, never directly editable
- When interests are updated, categories and interest_headline are regenerated
- add_by field: 'manual' (owner added) or 'email' (self-signup)
- auto_send_emails: True means emails are sent automatically (default for both manual and email signups)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# Revision identifiers
revision: str = 'colleague_email_log'
down_revision: Union[str, None] = 'add_colleague_onboarding'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add interest_headline column to colleagues table
    # This caches the LLM-generated headline like "Computer Vision, Medical Imaging, and RAG"
    op.add_column(
        'colleagues',
        sa.Column('interest_headline', sa.String(255), nullable=True)
    )
    
    # 2. Create colleague_email_log table for tracking outbound emails
    op.create_table(
        'colleague_email_log',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('colleague_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('colleagues.id', ondelete='CASCADE'), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Email details
        sa.Column('message_id', sa.String(255), nullable=True),  # SMTP/Gmail message ID
        sa.Column('subject', sa.String(500), nullable=False),
        sa.Column('snippet', sa.Text, nullable=True),  # First ~100 chars for preview
        
        # Type categorization
        sa.Column('email_type', sa.String(50), nullable=False, server_default='paper_recommendation'),
        
        # Paper reference (if email is about a specific paper)
        sa.Column('paper_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('papers.id', ondelete='SET NULL'), nullable=True),
        sa.Column('paper_arxiv_id', sa.String(50), nullable=True),
        
        # Timestamps
        sa.Column('sent_at', sa.DateTime(), nullable=False, server_default=sa.text('NOW()')),
        sa.Column('created_at', sa.DateTime(), nullable=True, server_default=sa.text('NOW()')),
        
        # Additional data (JSON) - named extra_data to avoid reserved word
        sa.Column('extra_data', postgresql.JSON, nullable=True),
    )
    
    # Create indexes for efficient queries
    op.create_index('ix_colleague_email_log_colleague_id', 'colleague_email_log', ['colleague_id'])
    op.create_index('ix_colleague_email_log_user_id', 'colleague_email_log', ['user_id'])
    op.create_index('ix_colleague_email_log_sent_at', 'colleague_email_log', ['sent_at'])
    op.create_index('ix_colleague_email_log_email_type', 'colleague_email_log', ['email_type'])
    
    # 3. Fix any existing colleagues with null auto_send_emails to True (default)
    # This ensures the UI displays consistent state
    op.execute("""
        UPDATE colleagues 
        SET auto_send_emails = TRUE 
        WHERE auto_send_emails IS NULL
    """)


def downgrade() -> None:
    # Drop indexes
    op.drop_index('ix_colleague_email_log_email_type', 'colleague_email_log')
    op.drop_index('ix_colleague_email_log_sent_at', 'colleague_email_log')
    op.drop_index('ix_colleague_email_log_user_id', 'colleague_email_log')
    op.drop_index('ix_colleague_email_log_colleague_id', 'colleague_email_log')
    
    # Drop table
    op.drop_table('colleague_email_log')
    
    # Drop column
    op.drop_column('colleagues', 'interest_headline')
