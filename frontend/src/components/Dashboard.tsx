import { useCallback, useEffect, useMemo, useState } from 'react';
import { keepPreviousData, useQueries, useQuery } from '@tanstack/react-query';
import {
  AlertCircle,
  ArrowLeft,
  Database,
  LoaderCircle,
  LogOut,
  RefreshCw,
  Route,
  ScanLine,
} from 'lucide-react';
import { FLIGHT_TABLES, RELATED_TABLES, TABLE_LABELS, type Api } from '../api';
import { FlightMap } from './FlightMap';
import DataTable from './DataTable';
import PassengerAvatar from './PassengerAvatar';
import {
  formatEvidence,
  formatValue,
  getPassengerId,
  getPassengerName,
  type Row,
} from './data';

type Props = { api: Api; onDisconnect: () => void };

const CLEANED_FLIGHTS_TABLE = 'flights_after';

function readPassengerRoute(): string | null {
  const id = new URLSearchParams(window.location.hash.slice(1)).get(
    'passenger',
  );
  return id?.trim() ? id : null;
}

function ErrorState({
  error,
  onRetry,
}: {
  error: unknown;
  onRetry: () => void;
}) {
  return (
    <div className="query-error" role="alert">
      <AlertCircle size={21} />
      <div>
        <strong>Не удалось получить данные</strong>
        <p>
          {error instanceof Error
            ? error.message
            : 'Проверьте подключение к API.'}
        </p>
      </div>
      <button className="button button-secondary" onClick={onRetry}>
        <RefreshCw size={14} /> Повторить
      </button>
    </div>
  );
}

function SessionControls({ onDisconnect }: { onDisconnect: () => void }) {
  return (
    <div className="workspace-controls">
      <a
        href="#overview"
        className="workspace-brand"
        aria-label="Найти шпионов — список шпионов"
      >
        <Route size={20} />
        <span>Найти шпионов</span>
      </a>
      <button
        className="icon-button"
        onClick={onDisconnect}
        aria-label="Отключиться и очистить данные"
        title="Отключиться"
      >
        <LogOut size={18} />
      </button>
    </div>
  );
}

function PassengerDetail({
  row,
  displayName,
}: {
  row: Row;
  displayName: string;
}) {
  const fields = Object.entries(row).filter(
    ([key]) =>
      ![
        'ФИО',
        'Идентификатор пассажира',
        'passenger_id',
        'PassengerDocument',
        'Пояснение',
      ].includes(key) && !key.endsWith('_info'),
  );
  return (
    <>
      <div className="profile-identity">
        <PassengerAvatar className="profile-avatar" />
        <div>
          <h3>{displayName}</h3>
          <span className="passenger-id">{getPassengerId(row)}</span>
        </div>
      </div>
      <dl className="profile-statistics">
        {fields.slice(0, 2).map(([key, value]) => (
          <div key={key}>
            <dt>{key === 'Количество вылетов' ? 'Вылеты, шт.' : key}</dt>
            <dd>{formatEvidence(key, value)}</dd>
          </div>
        ))}
      </dl>
      {row['Пояснение'] !== null && row['Пояснение'] !== undefined && (
        <div className="explanation">
          <div className="explanation-heading">
            <ScanLine size={16} />
            <h4>Почему запись выделена</h4>
          </div>
          <p>{formatValue(row['Пояснение'])}</p>
        </div>
      )}
      {fields.length > 2 && (
        <details className="feature-details">
          <summary>
            Аналитические признаки <span>{fields.length - 2}</span>
          </summary>
          <dl>
            {fields.slice(2).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd>{formatEvidence(key, value)}</dd>
              </div>
            ))}
          </dl>
        </details>
      )}
    </>
  );
}

export default function Dashboard({ api, onDisconnect }: Props) {
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState(10);
  const [selectedId, setSelectedId] = useState(readPassengerRoute);
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [selectedId]);
  useEffect(() => {
    const updateRoute = () => setSelectedId(readPassengerRoute());
    window.addEventListener('hashchange', updateRoute);
    window.addEventListener('popstate', updateRoute);
    return () => {
      window.removeEventListener('hashchange', updateRoute);
      window.removeEventListener('popstate', updateRoute);
    };
  }, []);
  const tables = useQuery({
    queryKey: ['tables'],
    queryFn: ({ signal }) => api.tables(signal),
  });
  const spies = useQuery({
    queryKey: ['spies', offset, limit],
    queryFn: ({ signal }) => api.spies(offset, limit, signal),
    placeholderData: keepPreviousData,
  });
  const detail = useQuery({
    queryKey: ['spy', selectedId],
    queryFn: ({ signal }) => api.spy(selectedId!, signal),
    enabled: selectedId !== null,
  });
  const availableRelated = useMemo(
    () => (tables.data ?? []).filter((name) => RELATED_TABLES.includes(name)),
    [tables.data],
  );
  const flightTables = useMemo(
    () => availableRelated.filter((name) => FLIGHT_TABLES.includes(name)),
    [availableRelated],
  );
  const { sources, pendingSources, failedSources } = useQueries({
    queries: flightTables.map((name) => ({
      queryKey: ['related', selectedId, name],
      queryFn: ({ signal }: { signal: AbortSignal }) =>
        api.related(selectedId!, name, signal),
      enabled: selectedId !== null,
    })),
    combine: (results) => ({
      sources: results.flatMap((query, index) =>
        query.data?.available
          ? [{ table: flightTables[index], rows: query.data.items }]
          : [],
      ),
      pendingSources: results.filter(
        (query) => query.isPending || query.isFetching,
      ).length,
      failedSources: results.flatMap((query, index) =>
        query.error || query.data?.available === false
          ? [{ refetch: query.refetch, table: flightTables[index] }]
          : [],
      ),
    }),
  });
  const related = useQuery({
    queryKey: ['related', selectedId, CLEANED_FLIGHTS_TABLE],
    queryFn: ({ signal }) =>
      api.related(selectedId!, CLEANED_FLIGHTS_TABLE, signal),
    enabled: selectedId !== null,
  });
  const select = useCallback((row: Row) => {
    window.location.hash = new URLSearchParams({
      passenger: getPassengerId(row),
    }).toString();
    setSelectedId(getPassengerId(row));
  }, []);
  const selected = detail.data;
  const selectedDisplayName = useMemo(() => {
    const listedRow = spies.data?.items.find(
      (row) => getPassengerId(row) === selectedId,
    );
    return getPassengerName(listedRow ?? selected ?? {});
  }, [selected, selectedId, spies.data?.items]);
  const displayedRelatedRows = useMemo(
    () =>
      (related.data?.items ?? []).map((row) =>
        Object.hasOwn(row, 'ФИО')
          ? { ...row, ФИО: selectedDisplayName }
          : row,
      ),
    [related.data?.items, selectedDisplayName],
  );
  const total = spies.data?.total;
  return (
    <div className="app-shell">
      <main
        className={`workspace ${selectedId ? 'passenger-page' : 'list-page'}`}
        id="overview"
      >
        {selectedId ? (
          <>
            <header className="page-heading">
              <div>
                <a href="#overview" className="back-to-list">
                  <ArrowLeft size={17} />
                  Назад к списку
                </a>
                <h1>История перелётов</h1>
              </div>
              <SessionControls onDisconnect={onDisconnect} />
            </header>
            <div className="passenger-layout">
              <section
                className="panel profile-panel"
                aria-labelledby="profile-heading"
              >
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">ПАССАЖИР</span>
                    <h2 id="profile-heading">Профиль</h2>
                  </div>
                  <ScanLine size={19} />
                </div>
                {detail.error ? (
                  <ErrorState
                    error={detail.error}
                    onRetry={() => void detail.refetch()}
                  />
                ) : selected ? (
                  <PassengerDetail
                    row={selected}
                    displayName={selectedDisplayName}
                  />
                ) : (
                  <div className="loading-state" role="status">
                    <LoaderCircle className="spinning" size={22} />
                    Загружаем профиль…
                  </div>
                )}
              </section>
              <section
                className="panel map-panel"
                aria-labelledby="flights-heading"
              >
                <div className="panel-heading">
                  <div>
                    <span className="eyebrow">МАРШРУТЫ</span>
                    <h2 id="flights-heading">Карта перелётов</h2>
                    <p>Перелёты из всех доступных таблиц</p>
                  </div>
                </div>
                {tables.error && (
                  <ErrorState
                    error={tables.error}
                    onRetry={() => void tables.refetch()}
                  />
                )}
                {pendingSources > 0 && (
                  <p className="source-status" role="status">
                    <LoaderCircle className="spinning" size={14} />
                    Загружаем источники: {pendingSources}
                  </p>
                )}
                {failedSources.length > 0 && (
                  <div className="query-error" role="alert">
                    <AlertCircle size={20} />
                    <div>
                      <strong>Карта неполная</strong>
                      <p>
                        Не загружены:{' '}
                        {failedSources
                          .map(({ table: name }) => TABLE_LABELS[name] ?? name)
                          .join(', ')}
                        .
                      </p>
                    </div>
                    <button
                      className="button button-secondary"
                      onClick={() =>
                        failedSources.forEach(({ refetch }) => void refetch())
                      }
                    >
                      <RefreshCw size={14} />
                      Повторить
                    </button>
                  </div>
                )}
                <FlightMap
                  key={selectedId}
                  sources={sources}
                  loading={pendingSources > 0 || tables.isPending}
                />
              </section>
            </div>
            <section
              className="panel related-panel source-records"
              aria-labelledby="related-heading"
            >
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">ПРОВЕРКА ИСТОЧНИКОВ</span>
                  <h2 id="related-heading">Исходные записи</h2>
                </div>
              </div>
              {related.error ? (
                <ErrorState
                  error={related.error}
                  onRetry={() => void related.refetch()}
                />
              ) : (
                <>
                  {related.data?.available === false && (
                    <p className="missing-table" role="status">
                      Таблица отсутствует. Проверьте локальный файл{' '}
                      {CLEANED_FLIGHTS_TABLE}.parquet на сервере.
                    </p>
                  )}
                  <div className="related-toolbar">
                    <span className="page-scope">
                      Все связанные строки ·{' '}
                      {formatValue(related.data?.total ?? 0)}
                    </span>
                  </div>
                  <DataTable
                    rows={displayedRelatedRows}
                    total={related.data?.total ?? 0}
                    loading={related.isFetching}
                    emptyMessage="В этой таблице нет строк, связанных с выбранным пассажиром."
                  />
                </>
              )}
            </section>
          </>
        ) : (
          <>
            <header className="page-heading">
              <div>
                <span className="eyebrow">АНАЛИЗ ПЕРЕЛЁТОВ</span>
                <h1>Список шпионов</h1>
              </div>
              <SessionControls onDisconnect={onDisconnect} />
            </header>
            <section
              className="summary-grid"
              aria-label="Состояние набора данных"
            >
              <div className="summary-card">
                <span className="summary-icon">
                  <ScanLine size={20} strokeWidth={1.5} />
                </span>
                <div>
                  <span className="summary-label">Записей в результатах</span>
                  <strong>
                    {total === undefined ? '—' : formatValue(total)}
                  </strong>
                </div>
              </div>
              <div className="summary-card">
                <span className="summary-icon">
                  <Database size={20} strokeWidth={1.5} />
                </span>
                <div>
                  <span className="summary-label">Подключено таблиц</span>
                  <strong>
                    {tables.data ? formatValue(tables.data.length) : '—'}
                  </strong>
                </div>
              </div>
              <div className="summary-card summary-context">
                <span className="summary-icon">
                  <Route size={20} strokeWidth={1.5} />
                </span>
                <div>
                  <span className="summary-label">Метод анализа</span>
                  <strong>LOF</strong>
                </div>
              </div>
            </section>
            {tables.error && (
              <ErrorState
                error={tables.error}
                onRetry={() => void tables.refetch()}
              />
            )}
            <section
              className="panel passengers-panel"
              aria-labelledby="passengers-heading"
            >
              <div className="panel-heading">
                <div>
                  <h2 id="passengers-heading">
                    Результаты анализа{' '}
                    <span className="heading-count">
                      {total === undefined ? '—' : formatValue(total)}
                    </span>
                  </h2>
                  <p>Откройте профиль, чтобы проследить историю перелётов</p>
                </div>
                <button
                  className="icon-button"
                  aria-label="Обновить результаты анализа"
                  disabled={spies.isFetching}
                  onClick={() => void spies.refetch()}
                >
                  <RefreshCw
                    className={spies.isFetching ? 'spinning' : ''}
                    size={17}
                  />
                </button>
              </div>
              <div className="table-toolbar">
                <label>
                  На странице
                  <select
                    aria-label="Количество пассажиров на странице"
                    value={limit}
                    onChange={(event) => {
                      setLimit(Number(event.target.value));
                      setOffset(0);
                    }}
                  >
                    {[10, 25, 50].map((size) => (
                      <option key={size} value={size}>
                        {size}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              {spies.error ? (
                <ErrorState
                  error={spies.error}
                  onRetry={() => void spies.refetch()}
                />
              ) : (
                <DataTable
                  rows={spies.data?.items ?? []}
                  total={total ?? 0}
                  pagination={{ offset, limit, onPageChange: setOffset }}
                  loading={spies.isFetching}
                  passengers
                  onSelect={select}
                  emptyMessage="В результатах анализа пока нет пассажиров."
                />
              )}
            </section>
          </>
        )}
      </main>
    </div>
  );
}
