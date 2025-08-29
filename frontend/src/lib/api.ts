import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8082'

export const api = axios.create({ baseURL: API_BASE })

export type MetaResponse = {
  table: string
  columns: any[]
  schema_map: Record<string, string>
  extra_numeric_columns: string[]
  db_path: string
  error?: string
}

export type PricesResponse = {
  dates: string[]
  open?: number[]
  high?: number[]
  low?: number[]
  close: number[]
  volume?: number[]
}

export type ForecastResponse = {
  dates: string[]
  observed: number[]
  predicted_path: number[]
  quantiles: { p10: number[]; p50: number[]; p90: number[] }
  diagnostics: any
}

export async function fetchMeta() {
  const { data } = await api.get<MetaResponse>('/api/meta')
  return data
}

export async function fetchStocks(q: string, limit = 20) {
  const { data } = await api.get<{ items: string[] }>('/api/stocks', { params: { q, limit } })
  return data.items
}

export async function fetchPrices(stock: string, start?: string, end?: string) {
  const { data } = await api.get<PricesResponse>('/api/prices', { params: { stock, start, end } })
  return data
}

export async function fetchForecast(params: {
  stock: string
  lookback_days: number
  horizon_days: number
  method: string
  feature_mode: string
  seed?: number
}) {
  const { data } = await api.get<ForecastResponse>('/api/forecast', { params })
  return data
}

