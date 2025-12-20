import { useState } from 'react';
import { BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Card } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select';
import { Checkbox } from './ui/checkbox';
import { Progress } from './ui/progress';
import { Upload, CheckCircle2 } from 'lucide-react';
import { useAppFilters, setAppFilters } from "@/state/appFilters.store";

const sentimentDistribution = [
  { name: 'Positive', value: 450, percentage: 45 },
  { name: 'Neutral', value: 350, percentage: 35 },
  { name: 'Negative', value: 200, percentage: 20 },
];

const sentimentByTeacher = [
  { teacher: 'Dr. García', positive: 80, neutral: 15, negative: 5 },
  { teacher: 'Prof. Martínez', positive: 65, neutral: 25, negative: 10 },
  { teacher: 'Dr. López', positive: 70, neutral: 20, negative: 10 },
  { teacher: 'Prof. Rodríguez', positive: 55, neutral: 30, negative: 15 },
  { teacher: 'Dr. Fernández', positive: 60, neutral: 25, negative: 15 },
];

const sampleData = [
  { id: 1, teacher: 'Dr. García', subject: 'Calculus I', rating: 4.5, comment: 'Excellent methodology' },
  { id: 2, teacher: 'Prof. Martínez', subject: 'Physics II', rating: 4.2, comment: 'Clear explanations' },
  { id: 3, teacher: 'Dr. López', subject: 'Programming', rating: 4.7, comment: 'Very helpful' },
  { id: 4, teacher: 'Prof. Rodríguez', subject: 'Chemistry', rating: 3.8, comment: 'Good class' },
  { id: 5, teacher: 'Dr. Fernández', subject: 'Mathematics', rating: 4.0, comment: 'Well organized' },
];

const COLORS = {
  positive: '#10B981',
  neutral: '#6B7280',
  negative: '#EF4444',
};

export function DataTab() {
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [dataLoaded, setDataLoaded] = useState(false);
  const [applyPreprocessing, setApplyPreprocessing] = useState(true);
  const [runSentiment, setRunSentiment] = useState(true);

  const handleLoadData = () => {
    setIsProcessing(true);
    setProgress(0);
    
    // Simulate processing
    const interval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 100) {
          clearInterval(interval);
          setIsProcessing(false);
          setDataLoaded(true);
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  };

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-white mb-2">Data</h2>
        <p className="text-gray-400">Data Ingestion and Analysis</p>
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Left Column - Data Ingestion */}
        <div className="space-y-6">
          {/* Upload Section */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Dataset Upload</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-2">Select File</label>
                <div className="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center hover:border-blue-500 transition-colors cursor-pointer">
                  <Upload className="w-8 h-8 text-gray-500 mx-auto mb-2" />
                  <p className="text-gray-400 text-sm">Click to upload or drag and drop</p>
                  <p className="text-gray-500 text-xs mt-1">CSV, XLSX (Max 10MB)</p>
                </div>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Dataset Name</label>
                <Input
                  placeholder="e.g., Evaluations_2025_1"
                  className="bg-[#0f1419] border-gray-700"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Semester</label>
                <Select defaultValue="2025-1">
                  <SelectTrigger className="bg-[#0f1419] border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    <SelectItem value="2025-1">2025-1</SelectItem>
                    <SelectItem value="2024-2">2024-2</SelectItem>
                    <SelectItem value="2024-1">2024-1</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Program</label>
                <Select defaultValue="engineering">
                  <SelectTrigger className="bg-[#0f1419] border-gray-700">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="bg-[#1a1f2e] border-gray-700">
                    <SelectItem value="engineering">Engineering</SelectItem>
                    <SelectItem value="sciences">Sciences</SelectItem>
                    <SelectItem value="arts">Arts</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-3 pt-2">
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="preprocessing"
                    checked={applyPreprocessing}
                    onCheckedChange={(checked) => setApplyPreprocessing(checked as boolean)}
                  />
                  <label htmlFor="preprocessing" className="text-sm text-gray-300 cursor-pointer">
                    Apply preprocessing
                  </label>
                </div>
                <div className="flex items-center space-x-2">
                  <Checkbox
                    id="sentiment"
                    checked={runSentiment}
                    onCheckedChange={(checked) => setRunSentiment(checked as boolean)}
                  />
                  <label htmlFor="sentiment" className="text-sm text-gray-300 cursor-pointer">
                    Run sentiment analysis (BETO)
                  </label>
                </div>
              </div>

              <Button
                onClick={handleLoadData}
                disabled={isProcessing}
                className="w-full bg-blue-600 hover:bg-blue-700"
              >
                {isProcessing ? 'Processing...' : 'Load and Process'}
              </Button>

              {isProcessing && (
                <div className="space-y-2">
                  <Progress value={progress} className="h-2 bg-gray-800" />
                  <p className="text-sm text-gray-400 text-center">{progress}% complete</p>
                </div>
              )}

              {dataLoaded && (
                <div className="flex items-center gap-2 text-green-400 text-sm">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>1,000 rows read, 998 valid</span>
                </div>
              )}
            </div>
          </Card>
        </div>

        {/* Right Column - Dataset Summary and Preview */}
        <div className="col-span-2 space-y-6">
          {/* Dataset Summary */}
          <Card className="bg-[#1a1f2e] border-gray-800 p-6">
            <h3 className="text-white mb-4">Dataset Summary</h3>
            
            {dataLoaded ? (
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-[#0f1419] p-4 rounded-lg">
                  <p className="text-gray-400 text-sm">Total Rows</p>
                  <p className="text-white text-2xl mt-1">1,000</p>
                </div>
                <div className="bg-[#0f1419] p-4 rounded-lg">
                  <p className="text-gray-400 text-sm">Columns</p>
                  <p className="text-white text-2xl mt-1">15</p>
                </div>
                <div className="bg-[#0f1419] p-4 rounded-lg">
                  <p className="text-gray-400 text-sm">Teachers</p>
                  <p className="text-white text-2xl mt-1">45</p>
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">No data loaded yet</p>
            )}

            {dataLoaded && (
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-800">
                      <th className="text-left text-gray-400 text-sm py-3 px-4">ID</th>
                      <th className="text-left text-gray-400 text-sm py-3 px-4">Teacher</th>
                      <th className="text-left text-gray-400 text-sm py-3 px-4">Subject</th>
                      <th className="text-left text-gray-400 text-sm py-3 px-4">Rating</th>
                      <th className="text-left text-gray-400 text-sm py-3 px-4">Comment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sampleData.map((row) => (
                      <tr key={row.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                        <td className="text-gray-300 text-sm py-3 px-4">{row.id}</td>
                        <td className="text-gray-300 text-sm py-3 px-4">{row.teacher}</td>
                        <td className="text-gray-300 text-sm py-3 px-4">{row.subject}</td>
                        <td className="text-gray-300 text-sm py-3 px-4">{row.rating}</td>
                        <td className="text-gray-300 text-sm py-3 px-4">{row.comment}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Sentiment Analysis Section */}
      {dataLoaded && runSentiment && (
        <div>
          <h3 className="text-white mb-4">Sentiment Analysis with BETO</h3>
          <div className="grid grid-cols-3 gap-6">
            {/* Sentiment Distribution */}
            <Card className="bg-[#1a1f2e] border-gray-800 p-6">
              <h4 className="text-white mb-4">Polarity Distribution</h4>
              <ResponsiveContainer width="100%" height={250}>
                <PieChart>
                  <Pie
                    data={sentimentDistribution}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percentage }) => `${name}: ${percentage}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {sentimentDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[entry.name.toLowerCase() as keyof typeof COLORS]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </Card>

            {/* Sentiment by Teacher */}
            <Card className="bg-[#1a1f2e] border-gray-800 p-6 col-span-2">
              <h4 className="text-white mb-4">Sentiment Distribution by Teacher</h4>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={sentimentByTeacher}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="teacher" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #374151' }}
                    labelStyle={{ color: '#fff' }}
                  />
                  <Legend />
                  <Bar dataKey="positive" stackId="a" fill={COLORS.positive} name="Positive" />
                  <Bar dataKey="neutral" stackId="a" fill={COLORS.neutral} name="Neutral" />
                  <Bar dataKey="negative" stackId="a" fill={COLORS.negative} name="Negative" />
                </BarChart>
              </ResponsiveContainer>
            </Card>
          </div>
        </div>
      )}
    </div>
  );
}