/**
 * FlowsPage (issue B5a) — the Workflows list page. Asserts the
 * loading/empty/error/list states, that toggling a flow calls
 * `setFlowEnabled` and refreshes the row, and that Run fires `runFlow`,
 * shows a "Workflow started" toast, and refetches the list.
 */
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import type { Flow } from '../services/api/flowsApi';
import { renderWithProviders } from '../test/test-utils';
import FlowsPage from './FlowsPage';

const listFlows = vi.hoisted(() => vi.fn());
const setFlowEnabled = vi.hoisted(() => vi.fn());
const runFlow = vi.hoisted(() => vi.fn());
vi.mock('../services/api/flowsApi', () => ({ listFlows, setFlowEnabled, runFlow }));

function makeFlow(overrides: Partial<Flow> = {}): Flow {
  return {
    id: 'flow-1',
    name: 'Daily digest',
    enabled: true,
    graph: { nodes: [], edges: [] },
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    last_run_at: null,
    last_status: null,
    require_approval: false,
    ...overrides,
  };
}

describe('FlowsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('shows a loading state while flows are being fetched', () => {
    listFlows.mockReturnValue(new Promise(() => {})); // never resolves
    renderWithProviders(<FlowsPage />);

    expect(screen.getByText('Loading workflows…')).toBeInTheDocument();
  });

  it('shows the empty state when there are no saved flows', async () => {
    listFlows.mockResolvedValue([]);
    renderWithProviders(<FlowsPage />);

    await waitFor(() => expect(screen.getByText('No workflows yet')).toBeInTheDocument());
    // The empty state omits a "Create" action (canvas ships in B5b).
    expect(screen.queryByRole('button', { name: /create/i })).not.toBeInTheDocument();
  });

  it('shows an error banner when the fetch fails', async () => {
    listFlows.mockRejectedValue(new Error('core unreachable'));
    renderWithProviders(<FlowsPage />);

    await waitFor(() =>
      expect(screen.getByText('Could not load workflows. Please try again.')).toBeInTheDocument()
    );
  });

  it('renders one row per saved flow', async () => {
    listFlows.mockResolvedValue([makeFlow(), makeFlow({ id: 'flow-2', name: 'Weekly report' })]);
    renderWithProviders(<FlowsPage />);

    await waitFor(() => expect(screen.getByText('Daily digest')).toBeInTheDocument());
    expect(screen.getByText('Weekly report')).toBeInTheDocument();
  });

  it('toggles a flow via setFlowEnabled and reflects the updated state', async () => {
    listFlows.mockResolvedValue([makeFlow({ enabled: true })]);
    setFlowEnabled.mockResolvedValue(makeFlow({ enabled: false }));
    renderWithProviders(<FlowsPage />);

    await waitFor(() => expect(screen.getByTestId('flow-toggle-flow-1')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('flow-toggle-flow-1'));

    expect(setFlowEnabled).toHaveBeenCalledWith('flow-1', false);
    await waitFor(() =>
      expect(screen.getByTestId('flow-status-flow-1')).toHaveTextContent('Paused')
    );
  });

  it('runs a flow, shows a "Workflow started" toast, and refetches the list', async () => {
    listFlows.mockResolvedValue([makeFlow()]);
    runFlow.mockResolvedValue({ output: null, pending_approvals: [], thread_id: 't1' });
    renderWithProviders(<FlowsPage />);

    await waitFor(() => expect(screen.getByTestId('flow-run-flow-1')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('flow-run-flow-1'));

    expect(runFlow).toHaveBeenCalledWith('flow-1');
    await waitFor(() => expect(screen.getByText('Workflow started')).toBeInTheDocument());
    // Loaded once on mount, once more on refetch after the run kicks off.
    await waitFor(() => expect(listFlows).toHaveBeenCalledTimes(2));
  });

  it('shows an error banner (without a toast) when runFlow rejects', async () => {
    listFlows.mockResolvedValue([makeFlow()]);
    runFlow.mockRejectedValue(new Error('flow disabled'));
    renderWithProviders(<FlowsPage />);

    await waitFor(() => expect(screen.getByTestId('flow-run-flow-1')).toBeInTheDocument());
    fireEvent.click(screen.getByTestId('flow-run-flow-1'));

    await waitFor(() => expect(screen.getByText('flow disabled')).toBeInTheDocument());
    expect(screen.queryByText('Workflow started')).not.toBeInTheDocument();
  });
});
