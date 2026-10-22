export type DataRow = Record<string, unknown>;

export interface Page<T = DataRow> {
  total: number;
  offset: number;
  limit: number;
  items: T[];
}

export interface RelatedRows {
  total: number;
  items: DataRow[];
  table: string;
  available: boolean;
  linkage: 'lineage' | 'representative_only';
}
