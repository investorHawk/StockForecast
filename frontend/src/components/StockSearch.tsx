import { useEffect, useMemo, useState } from 'react'
import { fetchStocks, StockItem } from '../lib/api'

type Props = {
  value: string
  onSelect: (code: string) => void
}

export default function StockSearch({ value, onSelect }: Props) {
  const [q, setQ] = useState(value || '')
  const [items, setItems] = useState<StockItem[]>([])

  useEffect(() => { setQ(value || '') }, [value])

  useEffect(() => {
    const id = setTimeout(async () => {
      if (q) {
        try {
          const items = await fetchStocks(q, 20)
          setItems(items)
        } catch { setItems([]) }
      } else { setItems([]) }
    }, 300)
    return () => clearTimeout(id)
  }, [q])

  return (
    <div className="space-y-2">
      <label className="text-sm font-medium">Stock Code</label>
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder="e.g., 0001, 7203, 218A"
        className="w-full border rounded px-3 py-2"
      />
      {items.length > 0 && (
        <div className="border rounded bg-white max-h-48 overflow-auto">
          {items.map((it) => (
            <div key={`${it.code}-${it.name}`} className="px-3 py-2 hover:bg-slate-100 cursor-pointer" onClick={() => onSelect(it.code)}>
              <div className="text-sm font-medium">{it.name || it.code}</div>
              <div className="text-xs text-slate-500">{it.code}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
