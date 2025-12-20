import { useState } from 'react';
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Input } from './ui/input';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Play, Award } from 'lucide-react';

const modelComparison = [
  { model: 'BM', accuracy: 0.82, f1: 0.79, precision: 0.81, recall: 0.78, time: 145 },
  { model: 'RBM', accuracy: 0.85, f1: 0.83, precision: 0.84, recall: 0.82, time: 132 },
  { model: 'DBM', accuracy: 0.89, f1: 0.87, precision: 0.88, recall: 0.86, time: 168 },
  { model: 'BM Pure', accuracy: 0.80, f1: 0.77, precision: 0.79, recall: 0.76, time: 138 },
  { model: 'RBM Pure', accuracy: 0.83, f1: 0.81, precision: 0.82, recall: 0.80, time: 125 },
];

const trainingLoss = [
  { epoch: 1, train: 0.65, validation: 0.68 },
  { epoch: 2, train: 0.52, validation: 0.56 },
  { epoch: 3, train: 0.43, validation: 0.48 },
  { epoch: 4, train: 0.37, validation: 0.42 },
  { epoch: 5, train: 0.32, validation: 0.39 },
  { epoch: 6, train: 0.28, validation: 0.36 },
  { epoch: 7, train: 0.25, validation: 0.34 },
  { epoch: 8, train: 0.23, validation: 0.33 },
];

const trainingAccuracy = [
  { epoch: 1, train: 0.68, validation: 0.65 },
  { epoch: 2, train: 0.74, validation: 0.71 },
  { epoch: 3, train: 0.78, validation: 0.75 },
  { epoch: 4, train: 0.82, validation: 0.79 },
  { epoch: 5, train: 0.85, validation: 0.82 },
  { epoch: 6, train: 0.87, validation: 0.84 },
  { epoch: 7, train: 0.89, validation: 0.86 },
  { epoch: 8, train: 0.90, validation: 0.87 },
];

const epochTime = [
  { epoch: 1, time: 22 },
  { epoch: 2, time: 21 },
  { epoch: 3, time: 21 },
  { epoch: 4, time: 20 },
  { epoch: 5, time: 21 },
  { epoch: 6, time: 20 },
  { epoch: 7, time: 21 },
  { epoch: 8, time: 20 },
];

const confusionMatrix = [
  { actual: 'High', predicted: 'High', value: 145 },
  { actual: 'High', predicted: 'Low', value: 22 },
  { actual: 'Low', predicted: 'High', value: 18 },
  { actual: 'Low', predicted: 'Low', value: 135 },
];

export function ModelsTab() {
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState('DBM');
  const [viewMode, setViewMode] = useState<'comparison' | 'details'>('comparison');

  const handleTrainModels = () => {
    setIsTraining(true);
    setTimeout(() => setIsTraining(false), 3000);
  };

  const bestModel = modelComparison.reduce((best, current) => 
    current.accuracy > best.accuracy ? current : best
  );

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-white mb-2">Models</h2>
          <p className="text-gray-400">Training and Comparison</p>
        </div>
        <Button
          onClick={handleTrainModels}
          disabled={isTraining}
          className="bg-blue-600 hover:bg-blue-700"
        >
          <Play className="w-4 h-4 mr-2" />
          {isTraining ? 'Training...' : 'Train All Models'}
        </Button>
      </div>

      {/* Configuration */}
      <Card className="bg-[#1a1f2e] border-gray-800 p-6">
        <h3 className="text-white mb-4">Training Configuration</h3>
        <div className="grid grid-cols-4 gap-4">
          <div>
            <label className="block text-sm text-gray-400 mb-2">Dataset</label>
            <Select defaultValue="dataset1">
              <SelectTrigger className="bg-[#0f1419] border-gray-700">
                <SelectValue />
              </SelectTrigger>
              <SelectContent className="bg-[#1a1f2e] border-gray-700">
                <SelectItem value="dataset1">Evaluations_2025_1</SelectItem>
                <SelectItem value="dataset2">Evaluations_2024_2</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Epochs</label>
            <Input
              type="number"
              defaultValue="10"
              className="bg-[#0f1419] border-gray-700"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Batch Size</label>
            <Input
              type="number"
              defaultValue="32"
              className="bg-[#0f1419] border-gray-700"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-400 mb-2">Learning Rate</label>
            <Input
              type="number"
              step="0.001"
              defaultValue="0.001"
              className="bg-[#0f1419] border-gray-700"
            />
          </div>
        </div>
      </Card>

      {/* Tabs */}
      <Tabs value={viewMode} onValueChange={(v) => setViewMode(v as 'comparison' | 'details')}>
        <TabsList className="bg-[#1a1f2e] border border-gray-800">
          <TabsTrigger value="comparison">Model Comparison</TabsTrigger>
          <TabsTrigger value="details">Best Model Details</TabsTrigger>
        </TabsList>

        <TabsContent value="comparison" className="space-y-6 mt-6">
          {/* Comparison Table */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Performance Metrics Comparison</h3>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-800">
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Model</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Accuracy</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">F1 Score</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Precision</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Recall</th>
                    <th className="text-left text-gray-400 text-sm py-3 px-4">Time (s)</th>
                  </tr>
                </thead>
                <tbody>
                  {modelComparison.map((model) => {
                    const isBest = model.model === bestModel.model;
                    return (
                      <tr
                        key={model.model}
                        className={`border-b border-gray-800/50 ${
                          isBest ? 'bg-blue-500/10' : 'hover:bg-gray-800/30'
                        }`}
                      >
                        <td className="text-gray-300 py-3 px-4 flex items-center gap-2">
                          {model.model}
                          {isBest && (
                            <Badge className="bg-blue-600 text-white">
                              <Award className="w-3 h-3 mr-1" />
                              Best
                            </Badge>
                          )}
                        </td>
                        <td className="text-gray-300 py-3 px-4">{(model.accuracy * 100).toFixed(1)}%</td>
                        <td className="text-gray-300 py-3 px-4">{(model.f1 * 100).toFixed(1)}%</td>
                        <td className="text-gray-300 py-3 px-4">{(model.precision * 100).toFixed(1)}%</td>
                        <td className="text-gray-300 py-3 px-4">{(model.recall * 100).toFixed(1)}%</td>
                        <td className="text-gray-300 py-3 px-4">{model.time}s</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Comparison Chart */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Accuracy Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelComparison}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="model" stroke="#9CA3AF" />
                <YAxis stroke="#9CA3AF" domain={[0.7, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
                />
                <Bar dataKey="accuracy" fill="#3B82F6" name="Accuracy" />
                <Bar dataKey="f1" fill="#06B6D4" name="F1 Score" />
              </BarChart>
            </ResponsiveContainer>
          </Card>
        </TabsContent>

        <TabsContent value="details" className="space-y-6 mt-6">
          {/* Selected Model Info */}
          <Card className="bg-gradient-to-r from-blue-600/20 to-cyan-600/20 border-blue-600/50 p-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-white mb-2">Selected Model: {bestModel.model}</h3>
                <p className="text-gray-300">
                  Accuracy: {(bestModel.accuracy * 100).toFixed(1)}% • 
                  F1 Score: {(bestModel.f1 * 100).toFixed(1)}% • 
                  Training Time: {bestModel.time}s
                </p>
              </div>
              <Award className="w-12 h-12 text-blue-400" />
            </div>
          </Card>

          {/* Training Curves */}
          <div className="grid grid-cols-2 gap-6">
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Training Loss</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={trainingLoss}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train" stroke="#3B82F6" strokeWidth={2} name="Training" />
                  <Line type="monotone" dataKey="validation" stroke="#F59E0B" strokeWidth={2} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </Card>

            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Training Accuracy</h3>
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={trainingAccuracy}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" domain={[0.6, 1]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Line type="monotone" dataKey="train" stroke="#10B981" strokeWidth={2} name="Training" />
                  <Line type="monotone" dataKey="validation" stroke="#06B6D4" strokeWidth={2} name="Validation" />
                </LineChart>
              </ResponsiveContainer>
            </Card>
          </div>

          {/* Time per Epoch and Confusion Matrix */}
          <div className="grid grid-cols-2 gap-6">
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Time per Epoch</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={epochTime}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="epoch" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Bar dataKey="time" fill="#8B5CF6" name="Time (seconds)" />
                </BarChart>
              </ResponsiveContainer>
            </Card>

            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h3 className="text-white mb-4">Confusion Matrix</h3>
              <div className="grid grid-cols-2 gap-4 mt-8">
                <div className="bg-green-500/20 border-2 border-green-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">True Positive</p>
                  <p className="text-white text-3xl mt-2">145</p>
                </div>
                <div className="bg-red-500/20 border-2 border-red-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">False Positive</p>
                  <p className="text-white text-3xl mt-2">18</p>
                </div>
                <div className="bg-red-500/20 border-2 border-red-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">False Negative</p>
                  <p className="text-white text-3xl mt-2">22</p>
                </div>
                <div className="bg-green-500/20 border-2 border-green-500/50 rounded-lg p-6 text-center">
                  <p className="text-gray-400 text-sm">True Negative</p>
                  <p className="text-white text-3xl mt-2">135</p>
                </div>
              </div>
              <div className="mt-4 text-sm text-gray-400 space-y-1">
                <p>Precision: {((145 / (145 + 18)) * 100).toFixed(1)}%</p>
                <p>Recall: {((145 / (145 + 22)) * 100).toFixed(1)}%</p>
                <p>Specificity: {((135 / (135 + 18)) * 100).toFixed(1)}%</p>
              </div>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
}
