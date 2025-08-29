type Props = {
  stock: string
  lookback: number
  horizon: number
  method: string
  feature_mode: string
  onChange: (p: { stock: string; lookback: number; horizon: number; method: string; feature_mode: string }) => void
}

export default function ParamsPanel({ stock, lookback, horizon, method, feature_mode, onChange }: Props) {
  return (
    <div className="space-y-2 bg-white rounded border p-3">
      <div className="font-medium">Parameters</div>
      <div className="grid grid-cols-2 gap-2">
        <label className="text-sm">Lookback</label>
        <input type="number" className="border rounded px-2 py-1" value={lookback} onChange={(e) => onChange({ stock, lookback: Number(e.target.value), horizon, method, feature_mode })} />
        <label className="text-sm">Horizon</label>
        <input type="number" className="border rounded px-2 py-1" value={horizon} onChange={(e) => onChange({ stock, lookback, horizon: Number(e.target.value), method, feature_mode })} />
        <label className="text-sm">Method</label>
        <select className="border rounded px-2 py-1" value={method} onChange={(e) => onChange({ stock, lookback, horizon, method: e.target.value, feature_mode })}>
          <option value="baseline">baseline</option>
          <option value="direct">direct</option>
          <option value="multistep">multistep</option>
          <option value="ensemble">ensemble</option>
        </select>
        <label className="text-sm">Features</label>
        <select className="border rounded px-2 py-1" value={feature_mode} onChange={(e) => onChange({ stock, lookback, horizon, method, feature_mode: e.target.value })}>
          <option value="auto">auto</option>
          <option value="ohlcv_only">ohlcv_only</option>
          <option value="all_numeric">all_numeric</option>
        </select>
      </div>
    </div>
  )
}

