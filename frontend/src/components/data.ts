export type Row = Record<string, unknown>;

export function getPassengerId(row: Row): string {
  return String(
    row['Идентификатор пассажира'] ??
      row.passenger_id ??
      row.PassengerDocument ??
      '',
  );
}

export function getPassengerName(row: Row): string {
  const name = row['ФИО'];
  return typeof name === 'string' && name.trim() ? name : 'Имя не указано';
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '—';
  if (typeof value === 'number')
    return new Intl.NumberFormat('ru-RU', { maximumFractionDigits: 4 }).format(
      value,
    );
  if (typeof value === 'object') return JSON.stringify(value);
  return String(value);
}

export function formatEvidence(key: string, value: unknown): string {
  if (key.startsWith('Доля') && typeof value === 'number')
    return new Intl.NumberFormat('ru-RU', {
      style: 'percent',
      maximumFractionDigits: 1,
    }).format(value);
  return formatValue(value);
}
