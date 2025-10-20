/**
* servicios/prediccion — Endpoints reales del router /prediccion
* - POST /prediccion/online
* - POST /prediccion/batch
*/
import api from "./apiClient";


export type OnlineInput = {
calificaciones: Record<string, number>;
comentario: string;
};


export type PrediccionOnlineRequest = {
job_id?: string | null;
family?: string; // default backend: "sentiment_desempeno"
input: OnlineInput;
};


export type PrediccionOnlineResponse = {
label_top: string;
scores: Record<string, number>;
sentiment?: string;
confidence?: number;
decision_rule?: Record<string, number> | null;
latency_ms: number;
correlation_id: string;
};


export type PrediccionBatchItem = {
id?: string;
calificaciones: Record<string, number>;
comentario: string;
};


export type PrediccionBatchResponse = {
batch_id: string;
summary: Record<string, any>;
sample: Array<Record<string, any>>;
artifact: string; // URL de descarga
correlation_id: string;
};


export async function online(req: PrediccionOnlineRequest) {
const { data } = await api.post<PrediccionOnlineResponse>("/prediccion/online", req);
return data;
}


export async function batch(file: File) {
const fd = new FormData();
fd.append("file", file);
const { data } = await api.post<PrediccionBatchResponse>("/prediccion/batch", fd);
return data;
}