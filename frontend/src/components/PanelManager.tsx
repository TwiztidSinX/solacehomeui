import React, { useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import GridLayout, { type Layout } from "react-grid-layout";
import "react-grid-layout/css/styles.css";
import "react-resizable/css/styles.css";

interface PanelConfig {
  id: string;
  title: string;
  component: React.ReactNode;
}

interface PanelManagerProps {
  panels: PanelConfig[];
  onPanelMoveOrResize?: (id: string) => void;
}

const DEFAULT_LAYOUT: Layout[] = [
  { i: "chat", x: 3, y: 0, w: 6, h: 14, minW: 3, maxW: 10 },
  { i: "chats", x: 0, y: 0, w: 3, h: 14, minW: 2, maxW: 5 },
  { i: "code", x: 9, y: 0, w: 3, h: 14, minW: 2, maxW: 8 },
  { i: "browser", x: 0, y: 14, w: 3, h: 12, minW: 2 },
  { i: "search", x: 4, y: 14, w: 3, h: 12, minW: 2 },
  { i: "parliament", x: 8, y: 14, w: 3, h: 12, minW: 2 },
  { i: "voice", x: 0, y: 26, w: 3, h: 10, minW: 2 },
  { i: "youtube", x: 4, y: 26, w: 3, h: 10, minW: 2 },
  { i: "tools", x: 8, y: 26, w: 3, h: 10, minW: 2 },
  { i: "agent-coding", x: 0, y: 36, w: 12, h: 14, minW: 3 },
];

const COLS = 12;
const STORAGE_KEY = "solace-panel-layout";

const rebalanceRow = (
  layout: Layout[],
  rowY: number,
  changedId?: string,
): Layout[] => {
  const next = layout.map((item) => ({ ...item }));
  const rowItems = next.filter((item) => item.y === rowY);
  if (rowItems.length === 0) return next;

  const changed = changedId
    ? rowItems.find((item) => item.i === changedId)
    : null;

  // If total width fits, just normalize x positions.
  const totalWidth = rowItems.reduce((sum, item) => sum + item.w, 0);
  if (totalWidth <= COLS) {
    let cursor = 0;
    rowItems
      .sort((a, b) => a.x - b.x)
      .forEach((item) => {
        item.x = cursor;
        item.y = rowY;
        cursor += item.w;
      });
    return next;
  }

  // Distribute width so the row always sums to COLS.
  if (changed) {
    const others = rowItems.filter((i) => i.i !== changed.i);
    const minOthers = others.reduce(
      (sum, i) => sum + (i.minW ?? 1),
      0,
    );
    const allowedForChanged = Math.max(1, COLS - minOthers);
    if (changed.w > allowedForChanged) changed.w = allowedForChanged;

    const remaining = COLS - changed.w;
    const othersWidth = others.reduce((sum, i) => sum + i.w, 0) || 1;
    others.forEach((item, index) => {
      const share = Math.max(
        item.minW ?? 1,
        Math.round((item.w / othersWidth) * remaining),
      );
      item.w = share;
      if (index === others.length - 1) {
        const currentTotal = rowItems.reduce((sum, i) => sum + i.w, 0);
        const delta = COLS - currentTotal;
        item.w = Math.max(item.minW ?? 1, item.w + delta);
      }
    });
  } else {
    // Scale all items proportionally.
    const factor = COLS / totalWidth;
    rowItems.forEach((item, index) => {
      const minW = item.minW ?? 1;
      const scaled = Math.max(minW, Math.round(item.w * factor));
      item.w = scaled;
      if (index === rowItems.length - 1) {
        const currentTotal = rowItems.reduce((sum, i) => sum + i.w, 0);
        const delta = COLS - currentTotal;
        item.w = Math.max(minW, item.w + delta);
      }
    });
  }

  // Re-pack x positions to keep the row tight.
  let cursor = 0;
  rowItems
    .sort((a, b) => a.x - b.x)
    .forEach((item) => {
      item.x = cursor;
      item.y = rowY;
      cursor += item.w;
    });
  return next;
};

export const PanelManager: React.FC<PanelManagerProps> = ({ panels, onPanelMoveOrResize }) => {
  const [layout, setLayout] = useState<Layout[]>(DEFAULT_LAYOUT);
  const [gridWidth, setGridWidth] = useState<number>(
    typeof window !== "undefined" ? Math.max(window.innerWidth, 1600) : 1600,
  );
  const [focusedPanelId, setFocusedPanelId] = useState<string | null>(null);

  useEffect(() => {
    const savedLayout = localStorage.getItem(STORAGE_KEY);
    if (savedLayout) {
      try {
        const parsed = JSON.parse(savedLayout) as Layout[];
        const sanitized = parsed.map((item) => ({
          ...item,
          static: false,
        }));
        setLayout(sanitized);
      } catch (error) {
        console.warn("Failed to parse saved layout, using default", error);
        setLayout(DEFAULT_LAYOUT);
      }
    } else {
      setLayout(DEFAULT_LAYOUT);
    }
  }, []);

  useEffect(() => {
    const handleResize = () =>
      setGridWidth(Math.max(window.innerWidth, 1600));
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    if (typeof document === "undefined") return;
    document.body.style.overflow = focusedPanelId ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [focusedPanelId]);

  useEffect(() => {
    setLayout((prevLayout) => {
      const existingIds = prevLayout.map((item) => item.i);
      const missingPanels = panels.filter(
        (panel) => !existingIds.includes(panel.id),
      );
      if (missingPanels.length === 0) return prevLayout;

      const nextY =
        prevLayout.reduce((max, item) => Math.max(max, item.y + item.h), 0) ||
        0;
      const additions = missingPanels.map((panel, index) => {
        const defaults = DEFAULT_LAYOUT.find((d) => d.i === panel.id);
        if (defaults) return { ...defaults };
        return {
          i: panel.id,
          x: (index * 4) % 12,
          y: nextY + index,
          w: 3,
          h: 12,
        };
      });

      const updatedLayout = [...prevLayout, ...additions];
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(updatedLayout),
      );
      return updatedLayout;
    });
  }, [panels]);

  const activeLayout = useMemo(
    () => layout.filter((item) => panels.some((panel) => panel.id === item.i)),
    [layout, panels],
  );

  const mergedLayout = useMemo(() => {
    if (activeLayout.length > 0) return activeLayout;
    return DEFAULT_LAYOUT.filter((item) =>
      panels.some((panel) => panel.id === item.i),
    );
  }, [activeLayout, panels]);

  const handleLayoutChange = (newLayout: Layout[]) => {
    setLayout((prevLayout) => {
      const otherItems = prevLayout.filter(
        (item) => !newLayout.find((layoutItem) => layoutItem.i === item.i),
      );
      const updatedLayout = [...newLayout, ...otherItems];
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(updatedLayout),
      );
      return updatedLayout;
    });
  };

  const handleResizeStop = (_layout: Layout[], _old: Layout, newItem: Layout) => {
    onPanelMoveOrResize?.(newItem.i);
    setLayout((prev) => {
      const next = rebalanceRow(prev, newItem.y, newItem.i);
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      return next;
    });
  };

  const toggleFocus = (panelId: string) => {
    setFocusedPanelId((prev) => (prev === panelId ? null : panelId));
  };

  const focusedPanel = useMemo(
    () => panels.find((p) => p.id === focusedPanelId),
    [focusedPanelId, panels],
  );

  return (
    <>
      {focusedPanelId && focusedPanel && createPortal(
        <div className="panel-focus-layer">
          <div
            className="panel-focus-overlay"
            onClick={() => toggleFocus(focusedPanelId)}
          />
          <div className="panel-focus-shell">
            <div
              className="panel-header"
              onDoubleClick={() => toggleFocus(focusedPanelId)}
              title="Double-click to exit focus"
            >
              <div className="panel-header-title">{focusedPanel.title}</div>
              <button
                className="panel-exit-focus"
                onClick={() => toggleFocus(focusedPanelId)}
                onMouseDown={(e) => e.stopPropagation()}
                onDoubleClick={(e) => e.stopPropagation()}
                type="button"
              >
                Exit focus
              </button>
            </div>
            <div className="panel-content">{focusedPanel.component}</div>
          </div>
        </div>,
        document.body,
      )}

      <GridLayout
        className="layout"
        layout={mergedLayout}
        cols={COLS}
        rowHeight={32}
        width={gridWidth}
        onLayoutChange={handleLayoutChange}
        draggableHandle=".panel-header"
        draggableCancel=".panel-content"
        isBounded={true}
        compactType={null}
        preventCollision={false}
        resizeHandles={['s', 'w', 'e', 'n', 'sw', 'nw', 'se', 'ne']}
        onResizeStop={handleResizeStop}
        onDragStop={(_layout, _old, newItem) => {
          onPanelMoveOrResize?.(newItem.i);
        }}
      >
        {panels.map((panel) => (
          <div
            key={panel.id}
            className={`panel-container${focusedPanelId === panel.id ? " panel-hidden-when-focused" : ""}`}
          >
            <div
              className="panel-header"
              onDoubleClick={() => toggleFocus(panel.id)}
              title="Double-click to focus"
            >
              <div className="panel-header-title">{panel.title}</div>
            </div>
            <div className="panel-content">{panel.component}</div>
          </div>
        ))}
      </GridLayout>
    </>
  );
};
