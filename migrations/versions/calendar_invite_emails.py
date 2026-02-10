"""Add calendar invite emails and inbound email replies tables.

Revision ID: 20260202_calendar_invites
Revises: add_bulk_action_fields
Create Date: 2026-02-02

This migration adds:
1. New columns to calendar_events for rescheduling support (ics_uid, sequence_number, etc.)
2. calendar_invite_emails table for tracking sent calendar invitations
3. inbound_email_replies table for processing user replies to calendar invites
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSON

# revision identifiers, used by Alembic.
revision = '20260202_calendar_invites'
down_revision = 'add_bulk_action_fields'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to calendar_events for rescheduling support
    op.add_column('calendar_events', sa.Column('ics_uid', sa.String(255), nullable=True))
    op.add_column('calendar_events', sa.Column('original_event_id', UUID(as_uuid=True), nullable=True))
    op.add_column('calendar_events', sa.Column('reschedule_note', sa.Text(), nullable=True))
    op.add_column('calendar_events', sa.Column('sequence_number', sa.Integer(), server_default='0', nullable=True))
    op.add_column('calendar_events', sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True))
    
    # Add self-referential foreign key for original_event_id
    op.create_foreign_key(
        'fk_calendar_events_original_event_id',
        'calendar_events', 'calendar_events',
        ['original_event_id'], ['id'],
        ondelete='SET NULL'
    )
    
    # Add index for ics_uid
    op.create_index('ix_calendar_ics_uid', 'calendar_events', ['ics_uid'], unique=False)
    
    # Create calendar_invite_emails table
    op.create_table(
        'calendar_invite_emails',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('calendar_event_id', UUID(as_uuid=True), sa.ForeignKey('calendar_events.id', ondelete='CASCADE'), nullable=False),
        sa.Column('email_id', UUID(as_uuid=True), sa.ForeignKey('emails.id', ondelete='SET NULL'), nullable=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        
        # Email threading identifiers
        sa.Column('message_id', sa.String(255), nullable=False, unique=True),
        sa.Column('thread_id', sa.String(255), nullable=True),
        sa.Column('in_reply_to', sa.String(255), nullable=True),
        
        sa.Column('recipient_email', sa.String(255), nullable=False),
        sa.Column('subject', sa.String(500), nullable=False),
        
        # ICS attachment details
        sa.Column('ics_uid', sa.String(255), nullable=False),
        sa.Column('ics_sequence', sa.Integer(), server_default='0'),
        sa.Column('ics_method', sa.String(20), server_default='REQUEST'),
        
        # Attribution
        sa.Column('triggered_by', sa.String(20), nullable=True, server_default='agent'),
        
        # Status
        sa.Column('status', sa.String(50), server_default='sent'),
        sa.Column('is_latest', sa.Boolean(), server_default='true'),
        
        sa.Column('sent_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    
    # Create indices for calendar_invite_emails
    op.create_index('ix_calendar_invite_email_calendar_event_id', 'calendar_invite_emails', ['calendar_event_id'])
    op.create_index('ix_calendar_invite_email_message_id', 'calendar_invite_emails', ['message_id'])
    op.create_index('ix_calendar_invite_email_thread_id', 'calendar_invite_emails', ['thread_id'])
    op.create_index('ix_calendar_invite_email_user_id', 'calendar_invite_emails', ['user_id'])
    op.create_index('ix_calendar_invite_email_ics_uid', 'calendar_invite_emails', ['ics_uid'])
    
    # Create inbound_email_replies table
    op.create_table(
        'inbound_email_replies',
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('user_id', UUID(as_uuid=True), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('original_invite_id', UUID(as_uuid=True), sa.ForeignKey('calendar_invite_emails.id', ondelete='CASCADE'), nullable=False),
        
        # Email headers
        sa.Column('message_id', sa.String(255), nullable=False, unique=True),
        sa.Column('in_reply_to', sa.String(255), nullable=True),
        sa.Column('references', sa.Text(), nullable=True),
        
        sa.Column('from_email', sa.String(255), nullable=False),
        sa.Column('subject', sa.String(500), nullable=True),
        sa.Column('body_text', sa.Text(), nullable=True),
        sa.Column('body_html', sa.Text(), nullable=True),
        
        # Agent interpretation
        sa.Column('intent', sa.String(50), nullable=True),
        sa.Column('extracted_datetime', sa.DateTime(), nullable=True),
        sa.Column('extracted_datetime_text', sa.String(500), nullable=True),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        
        # Processing status
        sa.Column('processed', sa.Boolean(), server_default='false'),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('processing_result', JSON, nullable=True),
        sa.Column('processing_error', sa.Text(), nullable=True),
        
        # Action taken
        sa.Column('action_taken', sa.String(50), nullable=True),
        sa.Column('new_event_id', UUID(as_uuid=True), sa.ForeignKey('calendar_events.id', ondelete='SET NULL'), nullable=True),
        
        sa.Column('received_at', sa.DateTime(), server_default=sa.text('now()')),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()')),
    )
    
    # Create indices for inbound_email_replies
    op.create_index('ix_inbound_reply_original_invite_id', 'inbound_email_replies', ['original_invite_id'])
    op.create_index('ix_inbound_reply_message_id', 'inbound_email_replies', ['message_id'])
    op.create_index('ix_inbound_reply_user_id', 'inbound_email_replies', ['user_id'])
    op.create_index('ix_inbound_reply_processed', 'inbound_email_replies', ['processed'])
    op.create_index('ix_inbound_reply_intent', 'inbound_email_replies', ['intent'])


def downgrade() -> None:
    # Drop inbound_email_replies table
    op.drop_index('ix_inbound_reply_intent', 'inbound_email_replies')
    op.drop_index('ix_inbound_reply_processed', 'inbound_email_replies')
    op.drop_index('ix_inbound_reply_user_id', 'inbound_email_replies')
    op.drop_index('ix_inbound_reply_message_id', 'inbound_email_replies')
    op.drop_index('ix_inbound_reply_original_invite_id', 'inbound_email_replies')
    op.drop_table('inbound_email_replies')
    
    # Drop calendar_invite_emails table
    op.drop_index('ix_calendar_invite_email_ics_uid', 'calendar_invite_emails')
    op.drop_index('ix_calendar_invite_email_user_id', 'calendar_invite_emails')
    op.drop_index('ix_calendar_invite_email_thread_id', 'calendar_invite_emails')
    op.drop_index('ix_calendar_invite_email_message_id', 'calendar_invite_emails')
    op.drop_index('ix_calendar_invite_email_calendar_event_id', 'calendar_invite_emails')
    op.drop_table('calendar_invite_emails')
    
    # Remove new columns from calendar_events
    op.drop_index('ix_calendar_ics_uid', 'calendar_events')
    op.drop_constraint('fk_calendar_events_original_event_id', 'calendar_events', type_='foreignkey')
    op.drop_column('calendar_events', 'updated_at')
    op.drop_column('calendar_events', 'sequence_number')
    op.drop_column('calendar_events', 'reschedule_note')
    op.drop_column('calendar_events', 'original_event_id')
    op.drop_column('calendar_events', 'ics_uid')
