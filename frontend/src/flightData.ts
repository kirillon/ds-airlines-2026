import type { FeatureCollection, LineString, Point } from 'geojson';
import type { DataRow } from './types';

export interface Airport {
  code: string;
  name: string;
  latitude: number;
  longitude: number;
}

export interface FlightSource {
  table: string;
  rows: DataRow[];
}

export const FLIGHT_SOURCE_STYLES: Record<
  string,
  { color: string; offset: number }
> = {
  flights: { color: '#185abd', offset: 0 },
  xml_flights: { color: '#007d9b', offset: 3 },
  yaml_flights: { color: '#7656b5', offset: -3 },
  tab: { color: '#c86b10', offset: 6 },
  xlsx: { color: '#b23e72', offset: -6 },
  flights_before: { color: '#55714c', offset: 9 },
  flights_after: { color: '#22366d', offset: -9 },
};

export function getFlightSourceStyle(table: string) {
  return FLIGHT_SOURCE_STYLES[table] ?? { color: '#506c8b', offset: 0 };
}

export function getFlightArrowImageId(table: string) {
  return `flight-arrow-${table || 'default'}`;
}

const CURVE_SEGMENTS = 32;
const ARROW_POSITION = 0.9;
const MAX_MERCATOR_LATITUDE = 85;
const MIN_ROUTE_DISTANCE = 1e-9;

function projectLatitude(latitude: number) {
  const clamped = Math.max(
    -MAX_MERCATOR_LATITUDE,
    Math.min(MAX_MERCATOR_LATITUDE, latitude),
  );
  const radians = (clamped * Math.PI) / 180;
  return (Math.log(Math.tan(Math.PI / 4 + radians / 2)) * 180) / Math.PI;
}

function unprojectLatitude(latitude: number) {
  return (
    (2 * Math.atan(Math.exp((latitude * Math.PI) / 180)) - Math.PI / 2) *
    (180 / Math.PI)
  );
}

function haveSameProjectedPosition(from: Airport, to: Airport) {
  const deltaLongitude =
    unwrapLongitude(to.longitude, from.longitude) - from.longitude;
  const deltaLatitude =
    projectLatitude(to.latitude) - projectLatitude(from.latitude);
  return Math.hypot(deltaLongitude, deltaLatitude) <= MIN_ROUTE_DISTANCE;
}

function buildCurvedRoute(
  from: Airport,
  to: Airport,
  center: number,
  offset: number,
) {
  const startLongitude = unwrapLongitude(from.longitude, center);
  const endLongitude = unwrapLongitude(to.longitude, startLongitude);
  const startLatitude = projectLatitude(from.latitude);
  const endLatitude = projectLatitude(to.latitude);
  const deltaLongitude = endLongitude - startLongitude;
  const deltaLatitude = endLatitude - startLatitude;
  const distance = Math.hypot(deltaLongitude, deltaLatitude);
  if (distance <= MIN_ROUTE_DISTANCE) {
    const position: [number, number] = [
      startLongitude,
      unprojectLatitude(startLatitude),
    ];
    return {
      coordinates: [position, position],
      arrowCoordinates: position,
      bearing: 0,
    };
  }
  const curvature = Math.max(0.06, Math.min(0.18, 0.12 + offset * 0.004));
  const bend = Math.min(18, distance * curvature);
  const controlLongitude =
    (startLongitude + endLongitude) / 2 - (deltaLatitude / distance) * bend;
  const controlLatitude =
    (startLatitude + endLatitude) / 2 + (deltaLongitude / distance) * bend;
  const pointAt = (position: number): [number, number] => {
    const remaining = 1 - position;
    const longitude =
      remaining * remaining * startLongitude +
      2 * remaining * position * controlLongitude +
      position * position * endLongitude;
    const latitude =
      remaining * remaining * startLatitude +
      2 * remaining * position * controlLatitude +
      position * position * endLatitude;
    return [longitude, unprojectLatitude(latitude)];
  };
  const tangentLongitude =
    2 *
    ((1 - ARROW_POSITION) * (controlLongitude - startLongitude) +
      ARROW_POSITION * (endLongitude - controlLongitude));
  const tangentLatitude =
    2 *
    ((1 - ARROW_POSITION) * (controlLatitude - startLatitude) +
      ARROW_POSITION * (endLatitude - controlLatitude));

  return {
    coordinates: Array.from({ length: CURVE_SEGMENTS + 1 }, (_, index) =>
      pointAt(index / CURVE_SEGMENTS),
    ),
    arrowCoordinates: pointAt(ARROW_POSITION),
    bearing:
      (Math.atan2(tangentLongitude, tangentLatitude) * 180) / Math.PI,
  };
}

const PAIRS = [
  ['departure_airport', 'arrival_airport'],
  ['origin', 'destination'],
  ['From', 'Dest'],
  ['Откуда', 'Куда'],
  ['Origin', 'Destination'],
  ['DepartureAirport', 'ArrivalAirport'],
] as const;

export function readAirport(value: unknown): Airport | null {
  if (!value || typeof value !== 'object') return null;
  const info = value as Record<string, unknown>;
  const { latitude, longitude, code, name } = info;
  if (
    typeof latitude !== 'number' ||
    typeof longitude !== 'number' ||
    !Number.isFinite(latitude) ||
    !Number.isFinite(longitude) ||
    Math.abs(latitude) > 90 ||
    Math.abs(longitude) > 180 ||
    typeof code !== 'string' ||
    !code.trim()
  )
    return null;
  return {
    code: code.trim(),
    name: typeof name === 'string' ? name : '',
    latitude,
    longitude,
  };
}

export function unwrapLongitude(longitude: number, reference: number): number {
  return longitude + 360 * Math.round((reference - longitude) / 360);
}

export function buildFlightData(rows: DataRow[]) {
  return buildSourceFlightData([{ table: '', rows }]);
}

export function buildSourceFlightData(sources: FlightSource[]) {
  const airports = new Map<string, Airport>();
  const pairs: [Airport, Airport, string][] = [];
  const summaries = sources.map(({ table, rows }) => ({
    table,
    rows: rows.length,
    routes: 0,
    missing: 0,
    sameAirport: 0,
    ...getFlightSourceStyle(table),
  }));
  let missing = 0;
  let sameAirport = 0;
  for (const [sourceIndex, source] of sources.entries()) {
    const summary = summaries[sourceIndex];
    for (const row of source.rows) {
      const fields = PAIRS.find(
        ([from, to]) =>
          from in row ||
          to in row ||
          `${from}_info` in row ||
          `${to}_info` in row,
      );
      const from = fields ? readAirport(row[`${fields[0]}_info`]) : null;
      const to = fields ? readAirport(row[`${fields[1]}_info`]) : null;
      for (const airport of [from, to]) {
        if (airport)
          airports.set(
            `${airport.code}:${airport.longitude}:${airport.latitude}`,
            airport,
          );
      }
      if (!from || !to) {
        missing++;
        summary.missing++;
        continue;
      }
      if (haveSameProjectedPosition(from, to)) {
        sameAirport++;
        summary.sameAirport++;
        continue;
      }
      pairs.push([from, to, source.table]);
      summary.routes++;
    }
  }
  const list = [...airports.values()].sort((a, b) =>
    a.code.localeCompare(b.code),
  );
  const sorted = list.map((a) => a.longitude).sort((a, b) => a - b);
  let gap = -1;
  let start = sorted[0] ?? 0;
  sorted.forEach((longitude, i) => {
    const next =
      sorted[(i + 1) % sorted.length] + (i === sorted.length - 1 ? 360 : 0);
    if (next - longitude > gap) {
      gap = next - longitude;
      start = next;
    }
  });
  const midpoint = start + (360 - Math.max(0, gap)) / 2;
  const center = ((((midpoint + 180) % 360) + 360) % 360) - 180;
  const routeData = pairs.map(([from, to, table]) => {
    const style = getFlightSourceStyle(table);
    return {
      table,
      style,
      arrowImage: getFlightArrowImageId(table),
      ...buildCurvedRoute(from, to, center, style.offset),
    };
  });
  const routes: FeatureCollection<LineString> = {
    type: 'FeatureCollection',
    features: routeData.map(({ table, style, coordinates }) => ({
      type: 'Feature',
      properties: { table, ...style },
      geometry: { type: 'LineString', coordinates },
    })),
  };
  const arrowheads: FeatureCollection<Point> = {
    type: 'FeatureCollection',
    features: routeData.map(
      ({ table, arrowImage, arrowCoordinates, bearing }) => ({
        type: 'Feature',
        properties: { table, icon: arrowImage, bearing },
        geometry: { type: 'Point', coordinates: arrowCoordinates },
      }),
    ),
  };
  const points: FeatureCollection<Point> = {
    type: 'FeatureCollection',
    features: list.map((airport) => ({
      type: 'Feature',
      properties: { code: airport.code },
      geometry: {
        type: 'Point',
        coordinates: [
          unwrapLongitude(airport.longitude, center),
          airport.latitude,
        ],
      },
    })),
  };
  const coordinates = [
    ...routes.features.flatMap((feature) => feature.geometry.coordinates),
    ...points.features.map((feature) => feature.geometry.coordinates),
  ];
  let bounds: [[number, number], [number, number]] | null = null;
  if (coordinates.length) {
    let west = Infinity;
    let east = -Infinity;
    let south = Infinity;
    let north = -Infinity;
    for (const [longitude, latitude] of coordinates) {
      west = Math.min(west, longitude);
      east = Math.max(east, longitude);
      const clampedLatitude = Math.max(-85, Math.min(85, latitude));
      south = Math.min(south, clampedLatitude);
      north = Math.max(north, clampedLatitude);
    }
    if (east - west > 360) {
      west = center - 180;
      east = center + 180;
    }
    bounds = [
      [west, south],
      [east, north],
    ];
  }
  return {
    routes,
    arrowheads,
    points,
    airports: list,
    missing,
    sameAirport,
    bounds,
    summaries,
    rowCount: summaries.reduce((total, source) => total + source.rows, 0),
  };
}
