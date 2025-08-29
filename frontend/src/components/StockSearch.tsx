import { useEffect, useMemo, useState } from 'react'
import { fetchStocks } from '../lib/api'

type Props = {
  value: string
  onSelect: (code: string) => void
}

export default function StockSearch({ value, onSelect }: Props) {
  const [q, setQ] = useState(value || '')
  const [items, setItems] = useState<string[]>([])

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
            <div key={it} className="px-3 py-2 hover:bg-slate-100 cursor-pointer" onClick={() => onSelect(it)}>
              {it}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

