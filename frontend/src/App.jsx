import { useEffect, useMemo, useState } from "react";
import axios from "axios";

import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";
import "./App.css";

const API_BASE_URL = import.meta.env.VITE_API_URL;

const api = axios.create({
  baseURL: API_BASE_URL,
});

const GEOJSON_URL = "/reference/milano-grid.geojson";

function MapUpdater({ location }) {
  const map = useMap();

  useEffect(() => {
    if (
      location &&
      Number.isFinite(location.centroid_latitude) &&
      Number.isFinite(location.centroid_longitude)
    ) {
      map.setView(
        [location.centroid_latitude, location.centroid_longitude],
        14
      );
    }
  }, [location, map]);

  return null;
}

function App() {
  const [page, setPage] = useState("overview");

  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [summaryError, setSummaryError] = useState("");

  const [gridId, setGridId] = useState("4821");
  const [gridData, setGridData] = useState(null);
  const [gridLoading, setGridLoading] = useState(false);
  const [gridError, setGridError] = useState("");

  const [hotspots, setHotspots] = useState([]);
  const [alerts, setAlerts] = useState([]);
  const [hotspotsLoading, setHotspotsLoading] = useState(false);
  const [hotspotsError, setHotspotsError] = useState("");

  const [displayLimit, setDisplayLimit] = useState(25);
  const [severityFilter, setSeverityFilter] = useState("ALL");

  const [geoJson, setGeoJson] = useState(null);
  const [geoJsonLoading, setGeoJsonLoading] = useState(false);
  const [geoJsonError, setGeoJsonError] = useState("");

  const [selectedGrid, setSelectedGrid] = useState(null);
  const [selectedLocation, setSelectedLocation] = useState(null);

  // =========================
  // RE5 - RISK STATE
  // =========================

  const [riskGridId, setRiskGridId] = useState("4821");
  const [riskFeatures, setRiskFeatures] = useState(null);
  const [riskFeaturesLoading, setRiskFeaturesLoading] = useState(false);
  const [riskFeaturesError, setRiskFeaturesError] = useState("");

  const [riskPrediction, setRiskPrediction] = useState(null);
  const [riskPredictionLoading, setRiskPredictionLoading] = useState(false);
  const [riskPredictionError, setRiskPredictionError] = useState("");

  // =========================
  // INITIAL SUMMARY
  // =========================

  useEffect(() => {
    async function loadSummary() {
      try {
        setSummaryLoading(true);
        setSummaryError("");

        const response = await api.get("/network/summary");
        setSummary(response.data);
      } catch (error) {
        setSummaryError(
          error.response?.data?.detail ||
            "Unable to load network summary. Check whether FastAPI is running."
        );
      } finally {
        setSummaryLoading(false);
      }
    }

    loadSummary();
  }, []);

  // =========================
  // GRID EXPLORER
  // =========================

  async function loadGrid(id = gridId) {
    const numericGridId = Number(id);

    if (
      !Number.isInteger(numericGridId) ||
      numericGridId < 1 ||
      numericGridId > 10000
    ) {
      setGridError("Grid ID must be between 1 and 10000.");
      setGridData(null);
      return;
    }

    try {
      setGridLoading(true);
      setGridError("");

      const response = await api.get(
        `/network/grid/${numericGridId}`
      );

      setGridData(response.data);
    } catch (error) {
      setGridData(null);

      setGridError(
        error.response?.data?.detail ||
          `Grid ${numericGridId} was not found.`
      );
    } finally {
      setGridLoading(false);
    }
  }

  function openGrid(id) {
    setGridId(String(id));
    setPage("grid");
    loadGrid(String(id));
  }

  // =========================
  // HOTSPOTS + ALERTS
  // =========================

  useEffect(() => {
    if (page !== "hotspots") {
      return;
    }

    async function loadHotspotsAndAlerts() {
      try {
        setHotspotsLoading(true);
        setHotspotsError("");

        const [hotspotResponse, alertResponse] = await Promise.all([
          api.get(`/network/hotspots?limit=${displayLimit}`),
          api.get("/network/alerts?limit=100"),
        ]);

        setHotspots(hotspotResponse.data?.alerts || []);
        setAlerts(alertResponse.data?.alerts || []);
      } catch (error) {
        setHotspotsError(
          error.response?.data?.detail ||
            "Unable to load hotspots and alerts."
        );
      } finally {
        setHotspotsLoading(false);
      }
    }

    loadHotspotsAndAlerts();
  }, [page, displayLimit]);

  // =========================
  // RE4 GEOJSON
  // =========================

  useEffect(() => {
    if (page !== "hotspots" || geoJson) {
      return;
    }

    async function loadGeoJson() {
      try {
        setGeoJsonLoading(true);
        setGeoJsonError("");

        const response = await fetch(GEOJSON_URL);

        if (!response.ok) {
          throw new Error("Unable to load Milan grid GeoJSON.");
        }

        const data = await response.json();
        setGeoJson(data);
      } catch (error) {
        setGeoJsonError(error.message);
      } finally {
        setGeoJsonLoading(false);
      }
    }

    loadGeoJson();
  }, [page, geoJson]);

  // =========================
  // RE4 FILTERING
  // =========================

  const filteredHotspots = useMemo(() => {
    if (severityFilter === "ALL") {
      return hotspots;
    }

    return hotspots.filter(
      (item) =>
        String(item.severity).toUpperCase() ===
        severityFilter
    );
  }, [hotspots, severityFilter]);

  const visibleHotspots = filteredHotspots.slice(
    0,
    Number(displayLimit)
  );

 const visibleGeoJson = useMemo(() => {
  if (!geoJson) {
    return null;
  }

  return geoJson;
}, [geoJson]);

  const hotspotByGrid = useMemo(() => {
    const map = new Map();

    hotspots.forEach((item) => {
      map.set(Number(item.grid_id), item);
    });

    return map;
  }, [hotspots]);

  function getGridVisualStatus(gridIdValue) {
    const item = hotspotByGrid.get(Number(gridIdValue));

    if (!item) {
      return "NORMAL";
    }

    const severity = String(item.severity).toUpperCase();

    if (severity === "HIGH") {
      return "HIGH";
    }

    if (severity === "MEDIUM") {
      return "ATTENTION";
    }

    return "NORMAL";
  }

  function getPolygonStyle(feature) {
    const cellId = Number(
      feature.properties?.cellId
    );

    const status = getGridVisualStatus(cellId);

    if (cellId === Number(selectedGrid)) {
      return {
        color: "#111827",
        weight: 4,
        dashArray: "8 4",
        fillOpacity: 0.55,
      };
    }

    if (status === "HIGH") {
      return {
        color: "#111827",
        weight: 3,
        dashArray: null,
        fillOpacity: 0.55,
      };
    }

    if (status === "ATTENTION") {
      return {
        color: "#374151",
        weight: 3,
        dashArray: "8 4",
        fillOpacity: 0.45,
      };
    }

    return {
      color: "#6b7280",
      weight: 1,
      dashArray: "3 5",
      fillOpacity: 0.12,
    };
  }

  async function handleFeatureClick(feature) {
    const cellId = Number(
      feature.properties?.cellId
    );

    if (!cellId) {
      return;
    }

    setSelectedGrid(cellId);

    try {
      const response = await api.get(
        `/network/grid/${cellId}/location`
      );

      setSelectedLocation(response.data);
    } catch {
      setSelectedLocation(null);
    }
  }

  function onEachFeature(feature, layer) {
    const cellId = Number(
      feature.properties?.cellId
    );

    layer.bindTooltip(`Grid ${cellId}`);

    layer.on({
      mouseover: () => {
        layer.setStyle({
          weight: 4,
        });
      },
      mouseout: () => {
        layer.setStyle(getPolygonStyle(feature));
      },
      click: () => {
        handleFeatureClick(feature);
      },
    });
  }

  // =========================
  // RE5 - LOAD FEATURES
  // =========================

  async function loadRiskFeatures(id = riskGridId) {
    const numericGridId = Number(id);

    if (
      !Number.isInteger(numericGridId) ||
      numericGridId < 1 ||
      numericGridId > 10000
    ) {
      setRiskFeatures(null);
      setRiskFeaturesError(
        "Grid ID must be between 1 and 10000."
      );
      return;
    }

    try {
      setRiskFeaturesLoading(true);
      setRiskFeaturesError("");
      setRiskPrediction(null);
      setRiskPredictionError("");

      const response = await api.get(
        `/network/grid/${numericGridId}/features`
      );

      setRiskFeatures(response.data);
    } catch (error) {
      setRiskFeatures(null);

      setRiskFeaturesError(
        error.response?.data?.detail ||
          `Features for Grid ${numericGridId} could not be loaded.`
      );
    } finally {
      setRiskFeaturesLoading(false);
    }
  }

  // =========================
  // RE5 - PREDICT RISK
  // =========================

  async function predictRisk() {
    if (!riskFeatures) {
      setRiskPredictionError(
        "Load grid features before requesting a prediction."
      );
      return;
    }

    try {
      setRiskPredictionLoading(true);
      setRiskPredictionError("");
      setRiskPrediction(null);

      const payload = {
        grid_id: riskFeatures.grid_id,
        feature_timestamp:
          riskFeatures.feature_timestamp,
        avg_activity: riskFeatures.avg_activity,
        activity_growth:
          riskFeatures.activity_growth,
        active_hours: riskFeatures.active_hours,
        peak_ratio: riskFeatures.peak_ratio,
        variability: riskFeatures.variability,
        internet_share:
          riskFeatures.internet_share,
      };

      const response = await api.post(
        "/network/predict-risk",
        payload
      );

      setRiskPrediction(response.data);
    } catch (error) {
      setRiskPredictionError(
        error.response?.data?.detail ||
          "Unable to generate risk prediction."
      );
    } finally {
      setRiskPredictionLoading(false);
    }
  }

  // =========================
  // RENDER
  // =========================

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>Network Intelligence NOC</h1>
          <p>Operational network monitoring dashboard</p>
        </div>
      </header>

      <nav className="navigation">
        <button
          className={page === "overview" ? "active" : ""}
          onClick={() => setPage("overview")}
        >
          Overview
        </button>

        <button
          className={page === "grid" ? "active" : ""}
          onClick={() => setPage("grid")}
        >
          Grid Explorer
        </button>

        <button
          className={page === "hotspots" ? "active" : ""}
          onClick={() => setPage("hotspots")}
        >
          Hotspots & Alerts
        </button>

        <button
          className={page === "risk" ? "active" : ""}
          onClick={() => setPage("risk")}
        >
          Risk
        </button>
      </nav>

      <main className="content">

        {/* ================================================= */}
        {/* OVERVIEW */}
        {/* ================================================= */}

        {page === "overview" && (
          <section>
            <div className="page-heading">
              <h2>Network Overview</h2>
              <p>
                Current network analytics supplied by the API.
              </p>
            </div>

            {summaryLoading && (
              <div className="status-banner">
                Loading network summary...
              </div>
            )}

            {summaryError && (
              <div className="status-banner error">
                {summaryError}
              </div>
            )}

            {summary && !summaryLoading && (
              <>
                <div className="status-banner success">
                  Network analytics available
                </div>

                <div className="cards">

                  <div className="card">
                    <span className="card-label">
                      Total Activity
                    </span>
                    <strong>
                      {summary.total_activity}
                    </strong>
                    <small>
                      Reported by analytics API
                    </small>
                  </div>

                  <div className="card">
                    <span className="card-label">
                      Active Grids
                    </span>
                    <strong>
                      {summary.active_grids}
                    </strong>
                    <small>
                      Grids with recorded activity
                    </small>
                  </div>

                  <div className="card">
                    <span className="card-label">
                      Peak Hour
                    </span>
                    <strong>
                      {String(summary.peak_hour).padStart(
                        2,
                        "0"
                      )}
                      :00
                    </strong>
                    <small>
                      Hour of highest network activity
                    </small>
                  </div>

                  <div className="card">
                    <span className="card-label">
                      Top Grid
                    </span>
                    <strong>
                      Grid {summary.top_grid}
                    </strong>
                    <small>
                      Highest activity grid
                    </small>
                  </div>

                </div>

                <div className="timestamp">
                  <strong>Reporting Timestamp</strong>
                  <span>{summary.as_of}</span>
                  <small>Timestamp supplied by API</small>
                </div>
              </>
            )}
          </section>
        )}

        {/* ================================================= */}
        {/* GRID EXPLORER */}
        {/* ================================================= */}

        {page === "grid" && (
          <section>
            <div className="page-heading">
              <h2>Grid Explorer</h2>
              <p>
                Inspect hourly network activity for a selected grid.
              </p>
            </div>

            <div className="control-row">
              <label>
                Grid ID
                <input
                  type="number"
                  min="1"
                  max="10000"
                  value={gridId}
                  onChange={(event) =>
                    setGridId(event.target.value)
                  }
                />
              </label>

              <button
                className="primary-button"
                onClick={() => loadGrid()}
                disabled={gridLoading}
              >
                {gridLoading ? "Loading..." : "Load Grid"}
              </button>
            </div>

            {gridError && (
              <div className="status-banner error">
                {gridError}
              </div>
            )}

            {gridData && (
              <>
                <div className="info-panel">
                  <div>
                    <strong>
                      Grid {gridData.grid_id}
                    </strong>
                  </div>

                  <div>
                    As of: {gridData.as_of}
                  </div>

                  <div>
                    Observations:{" "}
                    {gridData.points?.length || 0}
                  </div>
                </div>

                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>SMS In</th>
                        <th>SMS Out</th>
                        <th>Call In</th>
                        <th>Call Out</th>
                        <th>Internet Activity</th>
                        <th>Total Activity</th>
                      </tr>
                    </thead>

                    <tbody>
                      {gridData.points?.map(
                        (point, index) => (
                          <tr key={`${point.timestamp}-${index}`}>
                            <td>{point.timestamp}</td>
                            <td>{point.sms_in}</td>
                            <td>{point.sms_out}</td>
                            <td>{point.call_in}</td>
                            <td>{point.call_out}</td>
                            <td>
                              {point.internet_activity}
                            </td>
                            <td>
                              {point.total_activity}
                            </td>
                          </tr>
                        )
                      )}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </section>
        )}

        {/* ================================================= */}
        {/* HOTSPOTS */}
        {/* ================================================= */}

        {page === "hotspots" && (
          <section>
            <div className="page-heading">
              <h2>Hotspots & Alerts</h2>
              <p>
                Ranked network activity and alert information.
              </p>
            </div>

            {hotspotsError && (
              <div className="status-banner error">
                {hotspotsError}
              </div>
            )}

            <div className="filter-bar">

              <label>
                Display Limit
                <select
                  value={displayLimit}
                  onChange={(event) =>
                    setDisplayLimit(
                      Number(event.target.value)
                    )
                  }
                >
                  <option value={5}>5</option>
                  <option value={10}>10</option>
                  <option value={25}>25</option>
                  <option value={50}>50</option>
                  <option value={100}>100</option>
                </select>
              </label>

              <label>
                Severity
                <select
                  value={severityFilter}
                  onChange={(event) =>
                    setSeverityFilter(event.target.value)
                  }
                >
                  <option value="ALL">All</option>
                  <option value="HIGH">High</option>
                  <option value="MEDIUM">Medium</option>
                  <option value="LOW">Low</option>
                </select>
              </label>

            </div>

            <div className="legend">
              <span className="legend-item normal">
                NORMAL
              </span>

              <span className="legend-item attention">
                ATTENTION
              </span>

              <span className="legend-item high">
                HIGH
              </span>
            </div>

            <div className="section-block">
              <h3>Ranked Hotspots</h3>

              {hotspotsLoading && (
                <div className="status-banner">
                  Loading hotspots...
                </div>
              )}

              {!hotspotsLoading &&
                visibleHotspots.length === 0 && (
                  <div className="empty-state">
                    No hotspots match the selected filter.
                  </div>
                )}

              {!hotspotsLoading &&
                visibleHotspots.length > 0 && (
                  <div className="table-wrapper">
                    <table>
                      <thead>
                        <tr>
                          <th>Grid</th>
                          <th>Activity</th>
                          <th>Alert Type</th>
                          <th>Severity</th>
                          <th>Status</th>
                          <th>Timestamp</th>
                          <th></th>
                        </tr>
                      </thead>

                      <tbody>
                        {visibleHotspots.map((item) => (
                          <tr key={item.alert_id}>
                            <td>
                              <strong>
                                Grid {item.grid_id}
                              </strong>
                            </td>

                            <td>{item.value}</td>

                            <td>{item.alert_type}</td>

                            <td>
                              <span
                                className={`severity ${String(
                                  item.severity
                                ).toLowerCase()}`}
                              >
                                {item.severity}
                              </span>
                            </td>

                            <td>{item.status}</td>

                            <td>{item.timestamp}</td>

                            <td>
                              <button
                                className="small-button"
                                onClick={() =>
                                  openGrid(item.grid_id)
                                }
                              >
                                Inspect
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
            </div>

            <div className="section-block">
              <h3>Milan Network Grid</h3>

              {geoJsonLoading && (
                <div className="status-banner">
                  Loading grid map...
                </div>
              )}

              {geoJsonError && (
                <div className="status-banner error">
                  {geoJsonError}
                </div>
              )}

              {visibleGeoJson && (
                <div className="map-container">
                  <MapContainer
                    center={[
                      45.4642,
                      9.19,
                    ]}
                    zoom={11}
                    scrollWheelZoom={true}
                  >
                    <TileLayer
                      attribution='&copy; OpenStreetMap contributors'
                      url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                    />

                    <GeoJSON
                      key={`${visibleHotspots.length}-${selectedGrid}`}
                      data={visibleGeoJson}
                      style={getPolygonStyle}
                      onEachFeature={onEachFeature}
                    />

                    <MapUpdater
                      location={selectedLocation}
                    />
                  </MapContainer>
                </div>
              )}

              {selectedLocation && (
                <div className="selected-grid">
                  <h4>
                    Selected Grid {selectedLocation.grid_id}
                  </h4>

                  <p>
                    Centroid:{" "}
                    {selectedLocation.centroid_latitude},{" "}
                    {selectedLocation.centroid_longitude}
                  </p>

                  <p>
                    Polygon Reference:{" "}
                    {selectedLocation.polygon_reference}
                  </p>
                </div>
              )}
            </div>

            <div className="section-block">
              <h3>Recent Alerts</h3>

              {alerts.length === 0 ? (
                <div className="empty-state">
                  No alerts available.
                </div>
              ) : (
                <div className="table-wrapper">
                  <table>
                    <thead>
                      <tr>
                        <th>Grid</th>
                        <th>Type</th>
                        <th>Severity</th>
                        <th>Value</th>
                        <th>Status</th>
                        <th>Timestamp</th>
                      </tr>
                    </thead>

                    <tbody>
                      {alerts.map((item) => (
                        <tr key={item.alert_id}>
                          <td>Grid {item.grid_id}</td>
                          <td>{item.alert_type}</td>
                          <td>
                            <span
                              className={`severity ${String(
                                item.severity
                              ).toLowerCase()}`}
                            >
                              {item.severity}
                            </span>
                          </td>
                          <td>{item.value}</td>
                          <td>{item.status}</td>
                          <td>{item.timestamp}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </section>
        )}

        {/* ================================================= */}
        {/* RE5 - PREDICTIVE RISK */}
        {/* ================================================= */}

        {page === "risk" && (
          <section>
            <div className="page-heading">
              <h2>Predictive Risk</h2>
              <p>
                Evaluate the latest stored network features for a
                selected grid using the prediction API.
              </p>
            </div>

            <div className="risk-flow">

              {/* GRID SELECTION */}

              <div className="risk-control-panel">
                <label>
                  Grid ID
                  <input
                    type="number"
                    min="1"
                    max="10000"
                    value={riskGridId}
                    onChange={(event) =>
                      setRiskGridId(event.target.value)
                    }
                  />
                </label>

                <button
                  className="primary-button"
                  onClick={() =>
                    loadRiskFeatures()
                  }
                  disabled={riskFeaturesLoading}
                >
                  {riskFeaturesLoading
                    ? "Loading Features..."
                    : "Load Features"}
                </button>
              </div>

              {riskFeaturesError && (
                <div className="status-banner error">
                  {riskFeaturesError}
                </div>
              )}

              {/* FEATURES */}

              {riskFeatures && (
                <>
                  <div className="risk-header">
                    <div>
                      <span className="card-label">
                        Selected Grid
                      </span>
                      <strong>
                        Grid {riskFeatures.grid_id}
                      </strong>
                    </div>

                    <div>
                      <span className="card-label">
                        Feature Timestamp
                      </span>
                      <strong>
                        {riskFeatures.feature_timestamp}
                      </strong>
                    </div>

                    <div>
                      <span className="card-label">
                        Data Quality
                      </span>
                      <strong>
                        {riskFeatures.data_quality_status}
                      </strong>
                    </div>

                    <div>
                      <span className="card-label">
                        Freshness
                      </span>
                      <strong>
                        {riskFeatures.freshness_hours} hours
                      </strong>
                    </div>
                  </div>

                  <div className="section-block">
                    <h3>Network Features</h3>

                    <div className="feature-grid">

                      <div className="feature-card">
                        <span>Average Activity</span>
                        <strong>
                          {riskFeatures.avg_activity}
                        </strong>
                      </div>

                      <div className="feature-card">
                        <span>Activity Growth</span>
                        <strong>
                          {riskFeatures.activity_growth}
                        </strong>
                      </div>

                      <div className="feature-card">
                        <span>Active Hours</span>
                        <strong>
                          {riskFeatures.active_hours}
                        </strong>
                      </div>

                      <div className="feature-card">
                        <span>Peak Ratio</span>
                        <strong>
                          {riskFeatures.peak_ratio}
                        </strong>
                      </div>

                      <div className="feature-card">
                        <span>Variability</span>
                        <strong>
                          {riskFeatures.variability}
                        </strong>
                      </div>

                      <div className="feature-card">
                        <span>Internet Share</span>
                        <strong>
                          {riskFeatures.internet_share}
                        </strong>
                      </div>

                    </div>
                  </div>

                  {/* PREDICT */}

                  <div className="risk-action">
                    <button
                      className="primary-button predict-button"
                      onClick={predictRisk}
                      disabled={riskPredictionLoading}
                    >
                      {riskPredictionLoading
                        ? "Predicting..."
                        : "Predict Risk"}
                    </button>
                  </div>
                </>
              )}

              {riskPredictionError && (
                <div className="status-banner error">
                  {riskPredictionError}
                </div>
              )}

              {/* RESULT */}

              {riskPrediction && (
                <div className="section-block">
                  <h3>Prediction Result</h3>

                  <div className="prediction-card">

                    <div className="prediction-score">
                      <span>Risk Score</span>
                      <strong>
                        {riskPrediction.risk_score}
                      </strong>
                    </div>

                    <div className="prediction-level">
                      <span>Risk Level</span>
                      <strong>
                        {riskPrediction.risk_level}
                      </strong>
                    </div>

                    <div className="prediction-version">
                      <span>Model Version</span>
                      <strong>
                        {riskPrediction.model_version}
                      </strong>
                    </div>

                  </div>

                  <div className="explanation-panel">
                    <strong>Explanation</strong>
                    <p>
                      {riskPrediction.explanation_note}
                    </p>
                  </div>

                  {riskPrediction.model_version ===
                    "stub-v1" && (
                    <div className="stub-notice">
                      <strong>
                        Contract-first prediction endpoint
                      </strong>
                      <p>
                        This result is currently produced by
                        the API5 stub. The trained ML model will
                        replace the stub in ML5 without changing
                        the API contract.
                      </p>
                    </div>
                  )}
                </div>
              )}

              {!riskFeatures &&
                !riskFeaturesLoading &&
                !riskFeaturesError && (
                  <div className="empty-state">
                    Enter a grid ID and load its latest network
                    features to begin.
                  </div>
                )}

            </div>
          </section>
        )}

      </main>
    </div>
  );
}

export default App;