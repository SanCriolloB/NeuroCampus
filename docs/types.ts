// Tipos compartidos (referenciales, no compilados en FE/BE)
export type JobId = string;
export type ApiError = { error: string; code?: string };

export type TrainRequest = { dataset_id: string; config: { algo: 'RBM'; epochs: number; seed?: number } };
export type TrainAccepted = { job_id: JobId };

export type OnlinePredictionReq = { features: Record<string, number|string> };
export type OnlinePredictionRes = { prediction: string; score: number; model_id: string; explain?: Record<string, number> };