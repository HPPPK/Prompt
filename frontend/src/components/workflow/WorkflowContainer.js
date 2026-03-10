import React, { useState } from 'react';
import {
  workflowClarify,
  workflowConfirmSpec,
  workflowExecute,
  workflowValidate,
} from '../../api';
import ClarifyForm from './ClarifyForm';
import SpecEditor from './SpecEditor';
import WorkflowResult from './WorkflowResult';

export default function WorkflowContainer({ initialData, inputText }) {
  const [workflow, setWorkflow] = useState(initialData);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const sessionId = workflow?.session_id;

  const onClarify = async (answers) => {
    setLoading(true);
    setError('');
    try {
      const next = await workflowClarify(sessionId, answers);
      setWorkflow((prev) => ({ ...prev, ...next }));
    } catch (e) {
      setError(e.message || 'Clarify failed');
    } finally {
      setLoading(false);
    }
  };

  const onConfirmSpec = async (spec) => {
    setLoading(true);
    setError('');
    try {
      const next = await workflowConfirmSpec(sessionId, spec);
      setWorkflow((prev) => ({ ...prev, ...next, spec_draft: spec, spec }));
    } catch (e) {
      setError(e.message || 'Confirm spec failed');
    } finally {
      setLoading(false);
    }
  };

  const onExecute = async (executor, config) => {
    setLoading(true);
    setError('');
    try {
      const next = await workflowExecute(sessionId, executor, config);
      setWorkflow((prev) => ({ ...prev, ...next, route: { ...(prev.route || {}), selected_executor: executor } }));
    } catch (e) {
      setError(e.message || 'Execute failed');
    } finally {
      setLoading(false);
    }
  };

  const onValidate = async (autoRevise) => {
    setLoading(true);
    setError('');
    try {
      const next = await workflowValidate(sessionId, autoRevise);
      setWorkflow((prev) => ({ ...prev, ...next }));
    } catch (e) {
      setError(e.message || 'Validation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="workflow-shell">
      <div className="wf-meta">
        <span className="meta-tag">Workflow</span>
        <span className="meta-tag">task: {workflow?.task_type}</span>
        <span className="meta-tag">state: {workflow?.state}</span>
      </div>
      <div className="wf-user-input">{inputText}</div>

      {error && <div className="wf-error">{error}</div>}

      {workflow?.state === 'clarifying' && (
        <ClarifyForm
          schema={workflow?.clarify_form_schema}
          missingSlots={workflow?.missing_slots || []}
          missingSlotHints={workflow?.missing_slot_hints || {}}
          onSubmit={onClarify}
          loading={loading}
        />
      )}

      {workflow?.state === 'spec_ready' && (
        <SpecEditor spec={workflow?.spec_draft || workflow?.spec} onSubmit={onConfirmSpec} loading={loading} />
      )}

      {(workflow?.state === 'done' || workflow?.state === 'executing' || workflow?.state === 'validating') && (
        <WorkflowResult data={workflow} onExecute={onExecute} onValidate={onValidate} loading={loading} />
      )}
    </div>
  );
}
