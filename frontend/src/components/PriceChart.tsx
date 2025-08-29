import { Line } from 'react-chartjs-2'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js'
import { ForecastResponse, PricesResponse } from '../lib/api'

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Tooltip, Legend, Filler)

type Props = {
  prices: PricesResponse | null
  forecast: ForecastResponse | null
  loading?: boolean
}

export default function PriceChart({ prices, forecast, loading }: Props) {
  if (loading) return <div className="p-4 bg-white rounded border">Loading...</div>
  if (!prices || prices.dates.length === 0) return <div className="p-4 bg-white rounded border">No data</div>

  const histDates = prices.dates
  const histClose = prices.close
  const fcDates = forecast?.dates ?? []
  const p50 = forecast?.predicted_path ?? []
  const p10 = forecast?.quantiles?.p10 ?? []
  const p90 = forecast?.quantiles?.p90 ?? []

  const labels = [...histDates, ...fcDates]
  // pad historical series to the full labels length so Chart.js aligns datasets
  const histSeries = [...histClose, ...new Array(fcDates.length).fill(null as any)]
  const p50Series = new Array(histClose.length).fill(null as any).concat(p50)
  const p90Series = new Array(histClose.length).fill(null as any).concat(p90)
  const p10Series = new Array(histClose.length).fill(null as any).concat(p10)

  const data = {
    labels,
    datasets: [
      {
        label: 'Observed',
        data: histSeries,
        borderColor: 'rgba(30, 64, 175, 1)',
        backgroundColor: 'rgba(30, 64, 175, 0.2)',
        pointRadius: 0,
        borderWidth: 2,
      },
      {
        label: 'p90',
        data: p90Series,
        borderColor: 'rgba(34,197,94,0.3)',
        backgroundColor: 'rgba(34,197,94,0.15)',
        pointRadius: 0,
        borderWidth: 0,
      },
      {
        label: 'p10',
        data: p10Series,
        borderColor: 'rgba(239,68,68,0.3)',
        backgroundColor: 'rgba(34,197,94,0.15)',
        pointRadius: 0,
        borderWidth: 0,
        fill: 1,
      },
      {
        label: 'Forecast p50',
        data: p50Series,
        borderColor: 'rgba(234, 179, 8, 1)',
        backgroundColor: 'rgba(234, 179, 8, 0.2)',
        pointRadius: 0,
        borderDash: [4, 4],
        borderWidth: 2,
        spanGaps: true,
      },
    ],
  }

  const options: any = {
    responsive: true,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { position: 'top' as const } },
    scales: {
      x: { display: true },
      y: { display: true, ticks: { callback: (v: any) => String(v) } },
    },
  }

  return (
    <div className="p-2 bg-white rounded border">
      <Line data={data as any} options={options} />
    </div>
  )
}
