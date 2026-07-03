/**
 * FlowsPage — the Workflows list page (issue B5a).
 *
 * The discoverable hub for the `flows::` domain: lists every saved
 * `Flow` (name, enabled toggle, last-run status, Run button). This is NOT the
 * canvas (B5b ships flow authoring/editing) and NOT the chat agent-proposal
 * surface (B4) — just the top-level `/flows` list, reached via the
 * "Workflows" nav tab (see `config/navConfig.ts`).
 */
import createDebug from 'debug';
import { useCallback, useEffect, useState } from 'react';

import EmptyStateCard from '../components/EmptyStateCard';
import FlowListRow, { type FlowListRowBusy } from '../components/flows/FlowListRow';
import { ToastContainer } from '../components/intelligence/Toast';
import PanelPage from '../components/layout/PanelPage';
import { CenteredLoadingState, ErrorBanner } from '../components/ui/LoadingState';
import { useT } from '../lib/i18n/I18nContext';
import { type Flow, listFlows, runFlow, setFlowEnabled } from '../services/api/flowsApi';
import type { ToastNotification } from '../types/intelligence';

const log = createDebug('app:flows');

/** Which single row + action currently has a request in flight, if any. */
type BusyKey = `toggle:${string}` | `run:${string}`;

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export default function FlowsPage() {
  const { t } = useT();
  const [flows, setFlows] = useState<Flow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busyKey, setBusyKey] = useState<BusyKey | null>(null);
  const [toasts, setToasts] = useState<ToastNotification[]>([]);

  const addToast = useCallback((toast: Omit<ToastNotification, 'id'>) => {
    setToasts(prev => [...prev, { ...toast, id: `toast-${Date.now()}-${Math.random()}` }]);
  }, []);
  const removeToast = useCallback((id: string) => {
    setToasts(prev => prev.filter(item => item.id !== id));
  }, []);

  const loadFlows = useCallback(async () => {
    log('loading flows');
    setLoading(true);
    setError(null);
    try {
      const result = await listFlows();
      setFlows(result);
      log('loaded %d flows', result.length);
    } catch (err) {
      log('load failed: %o', err);
      setError(t('flows.page.loadError'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    void loadFlows();
  }, [loadFlows]);

  const handleToggle = useCallback(
    async (flow: Flow) => {
      if (busyKey) return;
      const key: BusyKey = `toggle:${flow.id}`;
      setBusyKey(key);
      setError(null);
      log('toggle: id=%s next=%s', flow.id, !flow.enabled);
      try {
        const updated = await setFlowEnabled(flow.id, !flow.enabled);
        setFlows(prev => prev.map(f => (f.id === updated.id ? updated : f)));
      } catch (err) {
        log('toggle failed: id=%s err=%o', flow.id, err);
        setError(errorMessage(err));
      } finally {
        setBusyKey(null);
      }
    },
    [busyKey]
  );

  const handleRun = useCallback(
    async (flow: Flow) => {
      if (busyKey) return;
      const key: BusyKey = `run:${flow.id}`;
      setBusyKey(key);
      setError(null);
      log('run: id=%s', flow.id);
      try {
        // Fire-and-forget: the caller doesn't wait for the run to finish,
        // just that it kicked off. The refetch below picks up the refreshed
        // `last_run_at` / `last_status` once the engine settles (or, for a
        // still-running flow, on the next manual refresh). Only refetch on
        // success — `loadFlows()` clears `error`, which would otherwise wipe
        // the failure banner set in the `catch` below.
        await runFlow(flow.id);
        addToast({ type: 'success', title: t('flows.list.runStarted') });
        await loadFlows();
      } catch (err) {
        log('run failed: id=%s err=%o', flow.id, err);
        setError(errorMessage(err));
      } finally {
        setBusyKey(null);
      }
    },
    [busyKey, addToast, loadFlows, t]
  );

  const busyFor = (flow: Flow): FlowListRowBusy => {
    if (busyKey === `toggle:${flow.id}`) return 'toggle';
    if (busyKey === `run:${flow.id}`) return 'run';
    return null;
  };

  return (
    <PanelPage
      testId="flows-page"
      title={t('flows.page.title')}
      description={t('flows.page.description')}>
      <div className="mx-auto w-full max-w-3xl space-y-4">
        {error && (
          <div data-testid="flows-error">
            <ErrorBanner message={error} />
          </div>
        )}

        {loading && <CenteredLoadingState label={t('flows.page.loading')} />}

        {!loading && flows.length === 0 && !error && (
          <EmptyStateCard
            icon={
              <svg
                className="h-7 w-7 text-primary-500"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}>
                <circle cx="5" cy="6" r="2" />
                <circle cx="5" cy="18" r="2" />
                <circle cx="19" cy="12" r="2" />
                <path strokeLinecap="round" d="M7 6h4a4 4 0 014 4M7 18h4a4 4 0 004-4" />
              </svg>
            }
            title={t('flows.page.emptyTitle')}
            description={t('flows.page.emptyDescription')}
          />
        )}

        {!loading && flows.length > 0 && (
          <div
            data-testid="flows-list"
            className="overflow-hidden rounded-2xl border border-line bg-surface">
            {flows.map(flow => (
              <FlowListRow
                key={flow.id}
                flow={flow}
                busy={busyFor(flow)}
                onToggle={f => void handleToggle(f)}
                onRun={f => void handleRun(f)}
              />
            ))}
          </div>
        )}

        {/* === B3b integration (wire after PR #4450 merges) ===
            "View runs" was pulled from `FlowListRow` for now — it would only
            store a `selectedFlowId` with nothing to show for it until the run
            inspector lands, which reads as a dead button. Once #4450 merges,
            re-add here as: track `selectedFlowId` state, list the flow's runs
            via listFlowRuns(flowId), and open the inspector
            (FlowRunInspectorDrawer, keyed by RUN id / thread_id, NOT flowId)
            for a chosen run:
        {selectedFlowId && (
          <FlowRunInspectorRunsForFlow
            flowId={selectedFlowId}
            onClose={() => setSelectedFlowId(null)}
          />
        )} */}
      </div>

      <ToastContainer notifications={toasts} onRemove={removeToast} />
    </PanelPage>
  );
}
