"""Add user_settings and processed_inbound_emails tables for inbox polling

Revision ID: 20260210_inbox_settings
Revises: cfb65c235999
Create Date: 2026-02-10

This migration adds:
1. user_settings table - stores inbox polling frequency and colleague join code
2. processed_inbound_emails table - tracks processed email IDs for idempotency
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID


# revision identifiers, used by Alembic.
revision: str = '20260210_inbox_settings'
down_revision: Union[str, None] = 'cfb65c235999'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create user_settings table
    op.create_table(
        'user_settings',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, unique=True),
        
        # Inbox polling configuration
        sa.Column('inbox_check_frequency_seconds', sa.Integer(), nullable=True),
        sa.Column('last_inbox_check_at', sa.DateTime(), nullable=True),
        sa.Column('inbox_check_enabled', sa.Boolean(), default=False),
        
        # Colleague join code (bcrypt hashed)
        sa.Column('colleague_join_code_hash', sa.String(255), nullable=True),
        sa.Column('colleague_join_code_updated_at', sa.DateTime(), nullable=True),
        
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    
    op.create_index('ix_user_settings_user_id', 'user_settings', ['user_id'])
    
    # Create processed_inbound_emails table
    op.create_table(
        'processed_inbound_emails',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Gmail message ID
        sa.Column('gmail_message_id', sa.String(255), nullable=False),
        
        # Processing metadata
        sa.Column('email_type', sa.String(50), nullable=False),
        sa.Column('processed_at', sa.DateTime(), server_default=sa.func.now()),
        sa.Column('processing_result', sa.String(50), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        
        # Original email metadata
        sa.Column('from_email', sa.String(255), nullable=True),
        sa.Column('subject', sa.String(500), nullable=True),
        
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now()),
        
        sa.UniqueConstraint('user_id', 'gmail_message_id', name='uq_processed_email_user_message'),
    )
    
    op.create_index('ix_processed_email_user_id', 'processed_inbound_emails', ['user_id'])
    op.create_index('ix_processed_email_gmail_id', 'processed_inbound_emails', ['gmail_message_id'])


def downgrade() -> None:
    op.drop_index('ix_processed_email_gmail_id', table_name='processed_inbound_emails')
    op.drop_index('ix_processed_email_user_id', table_name='processed_inbound_emails')
    op.drop_table('processed_inbound_emails')
    
    op.drop_index('ix_user_settings_user_id', table_name='user_settings')
    op.drop_table('user_settings')
