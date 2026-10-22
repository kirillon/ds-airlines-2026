import { useMemo } from 'react';
import { ChevronLeft, ChevronRight, ArrowUpRight, Inbox } from 'lucide-react';
import {
  flexRender,
  getCoreRowModel,
  useReactTable,
  type ColumnDef,
} from '@tanstack/react-table';
import {
  formatValue,
  getPassengerId,
  getPassengerName,
  type Row,
} from './data';
import PassengerAvatar from './PassengerAvatar';

type Props = {
  rows: Row[];
  total: number;
  pagination?: {
    offset: number;
    limit: number;
    onPageChange: (offset: number) => void;
  };
  loading?: boolean;
  passengers?: boolean;
  selectedId?: string | null;
  onSelect?: (row: Row) => void;
  emptyMessage?: string;
};

type SelectionMeta = {
  selectedId?: string | null;
  onSelect?: (row: Row) => void;
};

const PASSENGER_COLUMNS: ColumnDef<Row>[] = [
  {
    id: 'passenger',
    header: 'Пассажир',
    cell: ({ row, table }) => {
      const selection = table.options.meta as SelectionMeta;
      return (
        <button
          className="passenger-button"
          onClick={() => selection.onSelect?.(row.original)}
          aria-pressed={selection.selectedId === getPassengerId(row.original)}
        >
          <PassengerAvatar className="passenger-list-avatar" />
          <span className="passenger-copy">
            <span className="passenger-name">
              {getPassengerName(row.original)}
            </span>
            <span className="passenger-id">{getPassengerId(row.original)}</span>
          </span>
        </button>
      );
    },
  },
  {
    id: 'flights',
    header: 'Вылеты, шт.',
    cell: ({ row }) => (
      <span className="numeric-cell">
        {formatValue(row.original['Количество вылетов'])}
      </span>
    ),
  },
  {
    id: 'lof',
    header: 'LOF',
    cell: ({ row }) => (
      <span className="score-pill">
        {formatValue(row.original['Оценка LOF'])}
      </span>
    ),
  },
  {
    id: 'select',
    header: '',
    cell: ({ row, table }) => (
      <button
        className="button button-secondary row-open"
        onClick={() =>
          (table.options.meta as SelectionMeta).onSelect?.(row.original)
        }
        aria-label={`Проследить: ${getPassengerName(row.original)}`}
      >
        Проследить <ArrowUpRight size={16} />
      </button>
    ),
  },
];

export default function DataTable({
  rows,
  total,
  pagination,
  loading = false,
  passengers = false,
  selectedId,
  onSelect,
  emptyMessage = 'В этой таблице пока нет строк.',
}: Props) {
  const rawColumns = useMemo<ColumnDef<Row>[]>(() => {
    const keys = [...new Set(rows.flatMap(Object.keys))].filter(
      (key) => !key.endsWith('_info'),
    );
    return keys.map((key) => ({
      id: key,
      header: key,
      cell: ({ row }) => (
        <span className="raw-cell" title={formatValue(row.original[key])}>
          {formatValue(row.original[key])}
        </span>
      ),
    }));
  }, [rows]);
  const table = useReactTable({
    data: rows,
    columns: passengers ? PASSENGER_COLUMNS : rawColumns,
    meta: { selectedId, onSelect },
    getCoreRowModel: getCoreRowModel(),
    manualPagination: true,
  });

  return (
    <div
      className={`data-table ${passengers ? 'passenger-table' : 'raw-table'}`}
      aria-busy={loading}
    >
      <div className="table-scroll">
        <table>
          <caption className="sr-only">
            {passengers
              ? 'Результаты анализа пассажиров'
              : 'Исходные строки выбранной таблицы'}
          </caption>
          <thead>
            {table.getHeaderGroups().map((group) => (
              <tr key={group.id}>
                {group.headers.map((header) => (
                  <th key={header.id} scope="col">
                    {flexRender(
                      header.column.columnDef.header,
                      header.getContext(),
                    )}
                  </th>
                ))}
              </tr>
            ))}
          </thead>
          <tbody>
            {table.getRowModel().rows.map((row) => (
              <tr
                key={row.id}
                className={
                  passengers && selectedId === getPassengerId(row.original)
                    ? 'selected-row'
                    : undefined
                }
              >
                {row.getVisibleCells().map((cell) => (
                  <td key={cell.id}>
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {rows.length === 0 && (
          <div className="table-empty" role="status">
            <Inbox size={24} strokeWidth={1.3} />
            <p>{loading ? 'Загружаем строки…' : emptyMessage}</p>
          </div>
        )}
      </div>
      {pagination && (
        <Pagination
          total={total}
          offset={pagination.offset}
          limit={pagination.limit}
          rowCount={rows.length}
          onPageChange={pagination.onPageChange}
          loading={loading}
        />
      )}
    </div>
  );
}

export function Pagination({
  total,
  offset,
  limit,
  rowCount,
  onPageChange,
  loading = false,
}: {
  total: number;
  offset: number;
  limit: number;
  rowCount: number;
  onPageChange: (offset: number) => void;
  loading?: boolean;
}) {
  const start = total === 0 ? 0 : Math.min(offset + 1, total);
  const end = Math.min(offset + rowCount, total);
  return (
    <div className="pagination">
      <span aria-live="polite">
        {loading
          ? 'Обновляем…'
          : `${start}–${end} из ${new Intl.NumberFormat('ru-RU').format(total)}`}
      </span>
      <div className="pagination-actions">
        <button
          className="icon-button"
          aria-label="Предыдущая страница"
          disabled={loading || offset === 0}
          onClick={() => onPageChange(Math.max(0, offset - limit))}
        >
          <ChevronLeft size={17} />
        </button>
        <button
          className="icon-button"
          aria-label="Следующая страница"
          disabled={loading || offset + limit >= total}
          onClick={() => onPageChange(offset + limit)}
        >
          <ChevronRight size={17} />
        </button>
      </div>
    </div>
  );
}
