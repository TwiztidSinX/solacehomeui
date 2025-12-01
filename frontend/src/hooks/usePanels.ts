import { useEffect, useState } from "react";
import { type LeftPanelMode } from "../components/LeftPanel";
import { type RightPanelMode } from "../components/RightPanel";

export const usePanels = () => {
  const [activeTab, setActiveTab] = useState("chat");
  const [leftPanelMode, setLeftPanelMode] =
    useState<LeftPanelMode>("chats");
  const [rightPanelMode, setRightPanelMode] =
    useState<RightPanelMode>("closed");
  const [leftPanelWidth, setLeftPanelWidth] = useState(280);
  const [rightPanelWidth, setRightPanelWidth] = useState(384);
  const [isResizingLeft, setIsResizingLeft] = useState(false);
  const [isResizingRight, setIsResizingRight] = useState(false);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (isResizingLeft) {
        const newWidth = Math.min(480, Math.max(220, e.clientX));
        setLeftPanelWidth(newWidth);
        return;
      }
      if (isResizingRight) {
        const newWidth = Math.min(
          1000,
          Math.max(320, window.innerWidth - e.clientX),
        );
        setRightPanelWidth(newWidth);
      }
    };

    const handleMouseUp = () => {
      if (isResizingLeft) setIsResizingLeft(false);
      if (isResizingRight) setIsResizingRight(false);
    };

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, [isResizingLeft, isResizingRight]);

  return {
    activeTab,
    setActiveTab,
    leftPanelMode,
    setLeftPanelMode,
    rightPanelMode,
    setRightPanelMode,
    leftPanelWidth,
    rightPanelWidth,
    startResizingLeft: () => setIsResizingLeft(true),
    startResizingRight: () => setIsResizingRight(true),
  };
};

export default usePanels;
