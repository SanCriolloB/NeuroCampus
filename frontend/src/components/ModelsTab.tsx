// ============================================================
// NeuroCampus — Models Tab (Orchestrator)
// 7 Sub-tabs: Resumen · Entrenamiento · Runs · Champion · Sweep · Bundle · Diagnóstico
// ============================================================
import { useState, useCallback } from 'react';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { motion } from 'motion/react';
import type { Family, RunRecord, ResolvedModel } from './models/mockData';
import { ModelContextHeader } from './models/ModelContextHeader';
import { SummarySubTab } from './models/SummarySubTab';
import { TrainingSubTab } from './models/TrainingSubTab';
import { RunsSubTab } from './models/RunsSubTab';
import { ChampionSubTab } from './models/ChampionSubTab';
import { SweepSubTab } from './models/SweepSubTab';
import { BundleSubTab } from './models/BundleSubTab';
import { DiagnosticSubTab } from './models/DiagnosticSubTab';

type SubTab = 'resumen' | 'entrenamiento' | 'runs' | 'champion' | 'sweep' | 'bundle' | 'diagnostico';

export function ModelsTab() {
  // Global context
  const [datasetId, setDatasetId] = useState('ds_2025_1');
  const [family, setFamily] = useState<Family>('sentiment_desempeno');
  const [resolvedModel, setResolvedModel] = useState<ResolvedModel | null>(null);

  // Sub-tab state
  const [activeSubTab, setActiveSubTab] = useState<SubTab>('resumen');

  // Extra runs created during this session (from training / sweep)
  const [sessionRuns, setSessionRuns] = useState<RunRecord[]>([]);

  // Navigation helpers: when clicking "Ver Run" from another tab
  const [navigateToRunId, setNavigateToRunId] = useState<string | null>(null);

  const handleNavigateToRun = useCallback((runId: string) => {
    setNavigateToRunId(runId);
    setActiveSubTab('runs');
  }, []);

  const handleUsePredictions = useCallback((runId: string) => {
    // In a real app, navigate to Predictions tab with params
    alert(`Navegar a Predictions con:\n  run_id=${runId}\n  dataset_id=${datasetId}\n  family=${family}`);
  }, [datasetId, family]);

  const handleTrainingComplete = useCallback((run: RunRecord) => {
    setSessionRuns(prev => [run, ...prev]);
  }, []);

  const handleSweepComplete = useCallback((runs: RunRecord[]) => {
    setSessionRuns(prev => [...runs, ...prev]);
  }, []);

  // When switching to runs tab and we had a navigate target, clear it after consumption
  const handleSubTabChange = (value: string) => {
    if (value !== 'runs') {
      setNavigateToRunId(null);
    }
    setActiveSubTab(value as SubTab);
  };

  return (
    <div className="p-6 lg:p-8 space-y-5 min-h-full">
      {/* Page Header */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h2 className="text-white mb-1">Modelos</h2>
        <p className="text-gray-400 text-sm">
          Entrenamiento, comparación y gestión de Redes de Boltzmann
        </p>
      </motion.div>

      {/* Global Context Header */}
      <ModelContextHeader
        datasetId={datasetId}
        family={family}
        onDatasetChange={setDatasetId}
        onFamilyChange={setFamily}
        resolvedModel={resolvedModel}
        onResolve={setResolvedModel}
      />

      {/* Sub-Tabs */}
      <Tabs value={activeSubTab} onValueChange={handleSubTabChange}>
        <TabsList className="bg-[#1a1f2e] border border-gray-800 flex-wrap h-auto gap-0.5 p-1">
          <TabsTrigger value="resumen" className="text-xs px-3 py-1.5">Resumen</TabsTrigger>
          <TabsTrigger value="entrenamiento" className="text-xs px-3 py-1.5">Entrenamiento</TabsTrigger>
          <TabsTrigger value="runs" className="text-xs px-3 py-1.5">Runs</TabsTrigger>
          <TabsTrigger value="champion" className="text-xs px-3 py-1.5">Champion</TabsTrigger>
          <TabsTrigger value="sweep" className="text-xs px-3 py-1.5">Sweep</TabsTrigger>
          <TabsTrigger value="bundle" className="text-xs px-3 py-1.5">Bundle / Artefactos</TabsTrigger>
          <TabsTrigger value="diagnostico" className="text-xs px-3 py-1.5">Diagnóstico</TabsTrigger>
        </TabsList>

        <TabsContent value="resumen" className="mt-5">
          <SummarySubTab
            family={family}
            datasetId={datasetId}
            onNavigateToRun={handleNavigateToRun}
            onUsePredictions={handleUsePredictions}
          />
        </TabsContent>

        <TabsContent value="entrenamiento" className="mt-5">
          <TrainingSubTab
            family={family}
            datasetId={datasetId}
            onTrainingComplete={handleTrainingComplete}
            onNavigateToRun={handleNavigateToRun}
            onUsePredictions={handleUsePredictions}
          />
        </TabsContent>

        <TabsContent value="runs" className="mt-5">
          <RunsSubTab
            family={family}
            datasetId={datasetId}
            extraRuns={sessionRuns}
            initialRunId={navigateToRunId}
            onUsePredictions={handleUsePredictions}
          />
        </TabsContent>

        <TabsContent value="champion" className="mt-5">
          <ChampionSubTab
            family={family}
            datasetId={datasetId}
            extraRuns={sessionRuns}
            onNavigateToRun={handleNavigateToRun}
            onUsePredictions={handleUsePredictions}
          />
        </TabsContent>

        <TabsContent value="sweep" className="mt-5">
          <SweepSubTab
            family={family}
            datasetId={datasetId}
            onNavigateToRun={handleNavigateToRun}
            onUsePredictions={handleUsePredictions}
            onSweepComplete={handleSweepComplete}
          />
        </TabsContent>

        <TabsContent value="bundle" className="mt-5">
          <BundleSubTab
            family={family}
            datasetId={datasetId}
          />
        </TabsContent>

        <TabsContent value="diagnostico" className="mt-5">
          <DiagnosticSubTab
            family={family}
            datasetId={datasetId}
          />
        </TabsContent>
      </Tabs>
    </div>
  );
}
