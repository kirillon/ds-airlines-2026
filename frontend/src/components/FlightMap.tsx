import { useEffect, useMemo, useRef, useState } from 'react';
import type { Map as LibreMap, Popup as LibrePopup } from 'maplibre-gl';
import 'maplibre-gl/dist/maplibre-gl.css';
import { TABLE_LABELS } from '../api';
import {
  buildSourceFlightData,
  getFlightArrowImageId,
} from '../flightData';
import type { FlightSource } from '../flightData';

function buildArrowImage(color: string) {
  const size = 28;
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas is unavailable');
  context.beginPath();
  context.moveTo(14, 2);
  context.lineTo(25, 15);
  context.lineTo(18, 15);
  context.lineTo(18, 26);
  context.lineTo(10, 26);
  context.lineTo(10, 15);
  context.lineTo(3, 15);
  context.closePath();
  context.fillStyle = color;
  context.strokeStyle = '#ffffff';
  context.lineWidth = 2;
  context.lineJoin = 'round';
  context.fill();
  context.stroke();
  return context.getImageData(0, 0, size, size);
}

export function FlightMap({
  sources,
  loading = false,
}: {
  sources: FlightSource[];
  loading?: boolean;
}) {
  const container = useRef<HTMLDivElement>(null);
  const instance = useRef<LibreMap | null>(null);
  const [hiddenSources, setHiddenSources] = useState<string[]>([]);
  const allData = useMemo(() => buildSourceFlightData(sources), [sources]);
  const data = useMemo(
    () =>
      hiddenSources.length
        ? buildSourceFlightData(
            sources.filter((source) => !hiddenSources.includes(source.table)),
          )
        : allData,
    [allData, sources, hiddenSources],
  );
  const [mapError, setMapError] = useState(false);

  useEffect(() => {
    if (!container.current || !data.bounds) return;
    let disposed = false;
    let map: LibreMap | undefined;
    let popup: LibrePopup | undefined;
    setMapError(false);
    void import('maplibre-gl')
      .then(({ Map, NavigationControl, Popup }) => {
        if (disposed || !container.current) return;
        map = new Map({
          container: container.current,
          maxPitch: 0,
          dragRotate: false,
          pitchWithRotate: false,
          style: {
            version: 8,
            sources: {
              basemap: {
                type: 'raster',
                tileSize: 256,
                tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
                attribution:
                  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
              },
              routes: { type: 'geojson', data: data.routes },
              arrowheads: { type: 'geojson', data: data.arrowheads },
              airports: { type: 'geojson', data: data.points },
            },
            layers: [
              {
                id: 'basemap',
                type: 'raster',
                source: 'basemap',
                paint: { 'raster-saturation': -1, 'raster-opacity': 0.65 },
              },
              {
                id: 'routes',
                type: 'line',
                source: 'routes',
                layout: { 'line-cap': 'round', 'line-join': 'round' },
                paint: {
                  'line-color': ['get', 'color'],
                  'line-width': 2.4,
                  'line-opacity': 0.86,
                },
              },
              {
                id: 'airports',
                type: 'circle',
                source: 'airports',
                paint: {
                  'circle-color': '#185abd',
                  'circle-radius': 5,
                  'circle-stroke-color': '#fff',
                  'circle-stroke-width': 2,
                },
              },
            ],
          },
          attributionControl: { compact: false },
          bounds: data.bounds!,
          fitBoundsOptions: { padding: 45, maxZoom: 7, duration: 0 },
        });
        instance.current = map;
        map.on('load', () => {
          if (!map) return;
          for (const source of data.summaries) {
            const imageId = getFlightArrowImageId(source.table);
            if (!map.hasImage(imageId))
              map.addImage(imageId, buildArrowImage(source.color));
          }
          map.addLayer(
            {
              id: 'route-arrowheads',
              type: 'symbol',
              source: 'arrowheads',
              layout: {
                'icon-image': ['get', 'icon'],
                'icon-size': 0.62,
                'icon-rotate': ['get', 'bearing'],
                'icon-rotation-alignment': 'map',
                'icon-pitch-alignment': 'map',
                'icon-allow-overlap': true,
                'icon-ignore-placement': true,
                'icon-keep-upright': false,
              },
            },
            'airports',
          );
        });
        map.touchZoomRotate.disableRotation();
        map.keyboard.disableRotation();
        map.addControl(
          new NavigationControl({ showCompass: false }),
          'top-right',
        );
        map.on('error', () => {
          if (!disposed) setMapError(true);
        });
        map.on('click', 'airports', (event) => {
          const code = event.features?.[0]?.properties?.code;
          const airport = data.airports.find((item) => item.code === code);
          if (!map || !airport) return;
          popup?.remove();
          popup = new Popup({ closeButton: true })
            .setLngLat(event.lngLat)
            .setText(
              `${airport.code}${airport.name ? ` — ${airport.name}` : ''}`,
            )
            .addTo(map);
        });
        map.on('mouseenter', 'airports', () => {
          if (map) map.getCanvas().style.cursor = 'pointer';
        });
        map.on('mouseleave', 'airports', () => {
          if (map) map.getCanvas().style.cursor = '';
        });
        map
          .getCanvas()
          .setAttribute('aria-label', 'Карта всех связанных перелётов');
      })
      .catch(() => {
        if (!disposed) setMapError(true);
      });
    return () => {
      disposed = true;
      instance.current = null;
      popup?.remove();
      map?.remove();
    };
  }, [data]);

  const resetView = () => {
    if (data.bounds)
      instance.current?.fitBounds(data.bounds, {
        padding: 45,
        maxZoom: 7,
        duration: 0,
      });
  };

  return (
    <section
      className="flight-map-section"
      aria-label="География перелётов"
      aria-busy={loading}
    >
      <div className="flight-map-header">
        {data.bounds && (
          <button
            type="button"
            className="button button-secondary"
            onClick={resetView}
          >
            Показать все маршруты
          </button>
        )}
      </div>
      {loading && <p role="status">Загружаем перелёты из таблиц…</p>}
      {data.bounds ? (
        <div
          ref={container}
          aria-label="Интерактивная карта перелётов"
          className="flight-map-canvas"
        />
      ) : (
        <p role="status">
          {allData.summaries.length > 0 &&
          allData.summaries.every((source) =>
            hiddenSources.includes(source.table),
          )
            ? 'Выберите источник, чтобы показать маршруты.'
            : 'Нет координат аэропортов для отображения карты.'}
        </p>
      )}
      {mapError && data.bounds && (
        <p role="status">
          Карта или подложка недоступна. Координаты аэропортов приведены ниже.
        </p>
      )}
      <fieldset className="source-legend">
        <legend>Источники на карте</legend>
        {allData.summaries.map((source) => (
          <label key={source.table} className="source-legend-item">
            <input
              type="checkbox"
              checked={!hiddenSources.includes(source.table)}
              onChange={(event) =>
                setHiddenSources((previous) =>
                  event.target.checked
                    ? previous.filter((table) => table !== source.table)
                    : [...previous, source.table],
                )
              }
            />
            <span
              className="source-legend-swatch"
              style={{ backgroundColor: source.color }}
              aria-hidden="true"
            />
            <span>{TABLE_LABELS[source.table] ?? source.table}</span>
            <span className="source-legend-count">
              {source.rows} строк · {source.routes} на карте
            </span>
          </label>
        ))}
      </fieldset>
      <p className="flight-map-summary" role="status">
        Выбрано строк: {data.rowCount}. Связей аэропортов:{' '}
        {data.routes.features.length}. Без пары координат: {data.missing}.
        {data.sameAirport > 0 && ` Совпадающие точки: ${data.sameAirport}.`}
      </p>
      <p className="flight-map-note">
        Один перелёт может присутствовать в нескольких таблицах. Цвет обозначает
        таблицу; совпадающие маршруты разведены по соседним дугам. Точка — аэропорт,
        стрелка показывает направление перелёта.
      </p>
      {data.airports.length > 0 && (
        <details>
          <summary>Аэропорты и координаты ({data.airports.length})</summary>
          <p style={{ lineHeight: 1.6, margin: '12px 0' }}>
            Координаты из API: широта, долгота (градусы).
          </p>
          <ul style={{ fontSize: 14, paddingLeft: 20 }}>
            {data.airports.map((airport) => (
              <li
                key={`${airport.code}:${airport.longitude}:${airport.latitude}`}
              >
                {airport.code}
                {airport.name ? ` — ${airport.name}` : ''}:{' '}
                {airport.latitude.toFixed(4)}, {airport.longitude.toFixed(4)}
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

export default FlightMap;
