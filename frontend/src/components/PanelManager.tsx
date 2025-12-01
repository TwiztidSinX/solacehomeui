import React, { useEffect, useMemo, useState } from "react";
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

const TOP_ROW_IDS = ["chats", "chat", "code"];
const COLS = 12;

const normalizeTopRow = (
  layout: Layout[],
  changedId?: string,
): Layout[] => {
  const next = layout.map((item) => ({ ...item }));
  const topItems = next.filter((item) => TOP_ROW_IDS.includes(item.i));
  if (topItems.length === 0) return next;

  const changed = changedId
    ? topItems.find((item) => item.i === changedId)
    : null;

  const sumMinOthers = topItems
    .filter((item) => item.i !== changedId)
    .reduce((sum, item) => sum + (item.minW ?? 1), 0);

  if (changed) {
    // Clamp the changed item so the others can keep their minimum space.
    const maxAllowed = Math.max(1, COLS - sumMinOthers);
    if (changed.w > maxAllowed) changed.w = maxAllowed;
  }

  // After clamping the changed item, redistribute the remaining width across the rest.
  const totalTopWidth = topItems.reduce((sum, item) => sum + item.w, 0);
  if (totalTopWidth > COLS) {
    const excess = totalTopWidth - COLS;
    const others = topItems.filter((item) => item.i !== changedId);
    const othersWidth = others.reduce((sum, item) => sum + item.w, 0) || 1;
    others.forEach((item, index) => {
      const share = Math.round((item.w / othersWidth) * excess);
      const newWidth = Math.max(item.minW ?? 1, item.w - share);
      item.w = newWidth;
      // If rounding left us with a mismatch, fix on last iteration.
      if (index === others.length - 1) {
        const correctedTotal =
          topItems.reduce((sum, i) => sum + i.w, 0) - (totalTopWidth - COLS);
        const delta = COLS - correctedTotal;
        item.w = Math.max(item.minW ?? 1, item.w + delta);
      }
    });
  }

  // Re-pack the top row so they stay in a single line without overlaps.
  let cursor = 0;
  TOP_ROW_IDS.forEach((id) => {
    const item = topItems.find((i) => i.i === id);
    if (!item) return;
    item.x = cursor;
    item.y = 0;
    cursor += item.w;
  });

  return next;
};

export const PanelManager: React.FC<PanelManagerProps> = ({ panels, onPanelMoveOrResize }) => {

  const [layout, setLayout] = useState<Layout[]>(DEFAULT_LAYOUT);
  const [gridWidth, setGridWidth] = useState<number>(
    typeof window !== "undefined" ? Math.max(window.innerWidth, 1600) : 1600,
  );

  useEffect(() => {
    const savedLayout = localStorage.getItem("solace-panel-layout");
    if (savedLayout) {
      try {
        const parsed = JSON.parse(savedLayout) as Layout[];
        const sanitized = parsed.map((item) => {
          const base = item.i === "chat" ? { ...item, static: false } : item;
          // Reapply sane minimums for top-row items to avoid wrap/drop.
          if (TOP_ROW_IDS.includes(item.i)) {
            const defaults = DEFAULT_LAYOUT.find((d) => d.i === item.i);
            return {
              ...base,
              minW: defaults?.minW ?? item.minW,
              maxW: defaults?.maxW ?? item.maxW,
            };
          }
          return base;
        });
        setLayout(normalizeTopRow(sanitized));
      } catch (error) {
        console.warn("Failed to parse saved layout, using default", error);
        setLayout(normalizeTopRow(DEFAULT_LAYOUT));
      }
    } else {
      setLayout(normalizeTopRow(DEFAULT_LAYOUT));
    }
  }, []);

  useEffect(() => {
    const handleResize = () =>
      setGridWidth(Math.max(window.innerWidth, 1600));
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

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
      const additions = missingPanels.map((panel, index) => ({
        i: panel.id,
        x: (index * 4) % 12,
        y: nextY + index,
        w: 3,
        h: 12,
      }));

      const updatedLayout = normalizeTopRow([...prevLayout, ...additions]);
      localStorage.setItem(
        "solace-panel-layout",
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
      // Only normalize if a top-row panel changed
      const topRowChanged = newLayout.some((newItem) => {
        if (!TOP_ROW_IDS.includes(newItem.i)) return false;
        const oldItem = prevLayout.find((p) => p.i === newItem.i);
        if (!oldItem) return false;
        return oldItem.w !== newItem.w || oldItem.x !== newItem.x;
      });

      const normalized = topRowChanged ? normalizeTopRow(newLayout) : newLayout;
      const otherItems = prevLayout.filter(
        (item) => !normalized.find((layoutItem) => layoutItem.i === item.i),
      );
      const updatedLayout = [...normalized, ...otherItems];
      localStorage.setItem(
        "solace-panel-layout",
        JSON.stringify(updatedLayout),
      );
      return updatedLayout;
    });
  };

  return (
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
      compactType="vertical"
      preventCollision={false}
      resizeHandles={['s', 'w', 'e', 'n', 'sw', 'nw', 'se', 'ne']}
      onResizeStop={(_layout, _old, newItem) => {
        onPanelMoveOrResize?.(newItem.i);
        setLayout((prev) => normalizeTopRow(prev, newItem.i));
      }}
      onDragStop={(_layout, _old, newItem) => {
        onPanelMoveOrResize?.(newItem.i);
      }}
    >
      {panels.map((panel) => (
        <div key={panel.id} className="panel-container">
          <div className="panel-header">{panel.title}</div>
          <div className="panel-content">{panel.component}</div>
        </div>
      ))}
    </GridLayout>
  );
};
