import type { DataRow, Page, RelatedRows } from './types';

export const TABLE_LABELS: Record<string, string> = {
  spies: 'Результаты LOF',
  passenger_sources: 'Связи источников',
  profiles: 'Профили · JSON',
  cards: 'Карты · JSON',
  flights: 'Перелёты · JSON',
  xml_profiles: 'Профили · XML',
  xml_cards: 'Карты · XML',
  xml_flights: 'Перелёты · XML',
  yaml_flights: 'Рейсы · YAML',
  yaml_passengers: 'Пассажиры · YAML',
  tab: 'Истории · TAB',
  xlsx: 'Посадочные талоны · XLSX',
  csv: 'Посадочные талоны · CSV',
  schedules: 'Расписание',
  flights_before: 'Перелёты до очистки',
  flights_after: 'Перелёты после очистки',
  passengers_before: 'Пассажиры до очистки',
  passengers_after: 'Пассажиры после очистки',
};

export const TABLE_NAMES = Object.keys(TABLE_LABELS);
export const RELATED_TABLES = TABLE_NAMES.filter(
  (table) => !['csv', 'schedules'].includes(table),
);
export const FLIGHT_TABLES = [
  'flights_after',
  'flights',
  'xml_flights',
  'yaml_flights',
  'tab',
  'xlsx',
  'flights_before',
];

export class ApiError extends Error {
  constructor(
    public readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

function normalizeBaseUrl(value: string): string {
  const base = value.trim().replace(/\/+$/, '') || '/api';
  if (base.startsWith('/') && !base.startsWith('//') && !/[?#]/.test(base))
    return base;
  try {
    const url = new URL(base);
    if (
      ['http:', 'https:'].includes(url.protocol) &&
      !url.username &&
      !url.password &&
      !url.search &&
      !url.hash
    ) {
      return url.href.replace(/\/+$/, '');
    }
  } catch {
    /* Validation message below applies to all invalid URL forms. */
  }
  throw new ApiError(
    0,
    'Укажите адрес HTTP(S) или относительный путь, например /api.',
  );
}

function getErrorMessage(status: number, body: unknown): string {
  if (status === 401 || status === 403)
    return 'Доступ отклонён. Проверьте API-ключ и подключитесь повторно.';
  if (status === 404) return 'Запись или таблица не найдена. Обновите данные.';
  if (status === 409)
    return 'Данные сейчас недоступны. Проверьте файлы на сервере.';
  const detail =
    body && typeof body === 'object' && 'detail' in body ? body.detail : null;
  if (typeof detail === 'string') {
    if (detail === 'Invalid or incomplete Parquet data')
      return 'Некорректные или неполные данные Parquet. Проверьте локальные файлы на сервере.';
    return detail;
  }
  if (Array.isArray(detail))
    return 'Сервер отклонил параметры запроса. Проверьте введённые значения.';
  return 'Не удалось выполнить запрос. Проверьте подключение к бекенду.';
}

export function createApi(baseUrl: string, key: string) {
  const base = normalizeBaseUrl(baseUrl);
  async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
    let response: Response;
    try {
      response = await fetch(`${base}${path}`, {
        ...init,
        headers: { 'X-API-Key': key, ...init.headers },
        cache: 'no-store',
      });
    } catch (error) {
      if (init.signal?.aborted) throw error;
      throw new ApiError(
        0,
        'Не удалось подключиться к бекенду. Проверьте адрес и доступность сервера.',
      );
    }
    const body: unknown = await response.json().catch(() => null);
    if (!response.ok)
      throw new ApiError(
        response.status,
        getErrorMessage(response.status, body),
      );
    if (body === null)
      throw new ApiError(
        response.status,
        'Сервер вернул неожиданный ответ. Проверьте адрес бекенда.',
      );
    return body as T;
  }
  return {
    tables: (signal?: AbortSignal) => request<string[]>('/tables', { signal }),
    spies: (offset = 0, limit = 50, signal?: AbortSignal) =>
      request<Page>(
        `/spies?${new URLSearchParams({ offset: String(offset), limit: String(limit) })}`,
        { signal },
      ),
    spy: (id: string, signal?: AbortSignal) =>
      request<DataRow>(
        `/spies/detail?${new URLSearchParams({ passenger_id: id })}`,
        { signal },
      ),
    related: (id: string, table: string, signal?: AbortSignal) =>
      request<RelatedRows>(
        `/spies/rows?${new URLSearchParams({ passenger_id: id, table })}`,
        { signal },
      ),
  };
}

export type Api = ReturnType<typeof createApi>;
