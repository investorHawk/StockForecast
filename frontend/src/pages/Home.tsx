import { useEffect, useMemo, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { fetchForecast, fetchMeta, fetchPrices, ForecastResponse, PricesResponse } from '../lib/api'
import StockSearch from '../components/StockSearch'
import ParamsPanel from '../components/ParamsPanel'
import PriceChart from '../components/PriceChart'

function useQuery() {
  return new URLSearchParams(useLocation().search)
}

export default function Home() {
  const q = useQuery()
  const navigate = useNavigate()
  const [meta, setMeta] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [prices, setPrices] = useState<PricesResponse | null>(null)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)

  const stock = q.get('stock') || '0001'
  const lookback = parseInt(q.get('lookback') || '126', 10)
  const horizon = parseInt(q.get('horizon') || '63', 10)
  const method = q.get('method') || 'ensemble'
  const feature_mode = q.get('feature_mode') || 'auto'

  useEffect(() => {
    fetchMeta().then(setMeta).catch(() => {})
  }, [])

  useEffect(() => {
    async function run() {
      setLoading(true)
      setError(null)
      try {
        const prices = await fetchPrices(stock)
        setPrices(prices)
        const fc = await fetchForecast({ stock, lookback_days: lookback, horizon_days: horizon, method, feature_mode })
        setForecast(fc)
      } catch (e: any) {
        setError(e?.response?.data?.detail || e?.message || 'Error')
      } finally {
        setLoading(false)
      }
    }
    if (stock) run()
  }, [stock, lookback, horizon, method, feature_mode])

  const onParamsChange = (next: { stock: string; lookback: number; horizon: number; method: string; feature_mode: string }) => {
    const search = new URLSearchParams({ stock: next.stock, lookback: String(next.lookback), horizon: String(next.horizon), method: next.method, feature_mode: next.feature_mode })
    navigate({ search: `?${search.toString()}` })
  }

  return (
    <div className="min-h-screen bg-slate-50">
      <div className="max-w-6xl mx-auto p-4 space-y-4">
        <h1 className="text-2xl font-semibold">JP Stock Forecast</h1>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="md:col-span-1 space-y-4">
            <StockSearch value={stock} onSelect={(s) => onParamsChange({ stock: s, lookback, horizon, method, feature_mode })} />
            <ParamsPanel
              stock={stock}
              lookback={lookback}
              horizon={horizon}
              method={method}
              feature_mode={feature_mode}
              onChange={onParamsChange}
            />
            {meta?.schema_map && (
              <div className="text-xs text-slate-600 p-2 bg-white rounded border">
                <div className="font-medium mb-1">Schema</div>
                <pre className="whitespace-pre-wrap">{JSON.stringify(meta.schema_map, null, 2)}</pre>
              </div>
            )}
          </div>
          <div className="md:col-span-3">
            {error && <div className="text-red-600 bg-red-50 border border-red-200 rounded p-3 mb-2">{error}</div>}
            <PriceChart prices={prices} forecast={forecast} loading={loading} />
          </div>
        </div>
      </div>
    </div>
  )
}

