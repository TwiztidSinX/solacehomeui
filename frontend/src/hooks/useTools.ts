import { useCallback, useEffect, useState } from "react";

export const useTools = () => {
  const [toolQuery, setToolQuery] = useState("");
  const [toolResults, setToolResults] = useState<
    Array<{ title: string; url: string; snippet: string }>
  >([]);
  const [toolLoading, setToolLoading] = useState(false);
  const [toolError, setToolError] = useState<string | null>(null);
  const [toolList, setToolList] = useState<
    Array<{ name: string; description?: string; parameters?: any }>
  >([]);
  const [selectedTool, setSelectedTool] = useState("");
  const [toolArgsText, setToolArgsText] = useState("{}");
  const [toolCallResult, setToolCallResult] = useState<string | null>(null);
  const [toolParamsByName, setToolParamsByName] = useState<Record<string, any>>(
    {},
  );

  const runToolSearch = useCallback(async () => {
    if (!toolQuery.trim()) {
      setToolError("Enter a query");
      return;
    }
    setToolError(null);
    setToolLoading(true);
    try {
      const res = await fetch(
        `http://localhost:8000/search?query=${encodeURIComponent(toolQuery.trim())}`,
      );
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      setToolResults(data.results || []);
    } catch (e: any) {
      setToolError(e.message || "Tool search failed");
    } finally {
      setToolLoading(false);
    }
  }, [toolQuery]);

  const fetchToolList = useCallback(async () => {
    try {
      const res = await fetch("http://localhost:8000/tools");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      const list = data.tools || [];
      setToolList(list);
      const map: Record<string, any> = {};
      list.forEach((t: any) => {
        if (t?.name) map[t.name] = t;
      });
      setToolParamsByName(map);
    } catch (e) {
      console.error("Failed to load tool list", e);
    }
  }, []);

  const runToolCall = useCallback(async () => {
    if (!selectedTool) {
      setToolError("Select a tool");
      return;
    }
    let argsObj: any = {};
    try {
      argsObj = toolArgsText ? JSON.parse(toolArgsText) : {};
      setToolError(null);
    } catch (e: any) {
      setToolError("Arguments must be valid JSON");
      return;
    }
    try {
      setToolLoading(true);
      const res = await fetch("http://localhost:8000/tool_call", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: selectedTool, arguments: argsObj }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setToolCallResult(
        typeof data.result === "string"
          ? data.result
          : JSON.stringify(data.result, null, 2),
      );
    } catch (e: any) {
      setToolError(e.message || "Tool call failed");
    } finally {
      setToolLoading(false);
    }
  }, [selectedTool, toolArgsText]);

  useEffect(() => {
    if (!selectedTool) return;
    const spec = toolParamsByName[selectedTool];
    if (!spec?.parameters?.properties) return;
    const props = spec.parameters.properties || {};
    const required: string[] = spec.parameters.required || [];
    const template: any = {};
    Object.keys(props).forEach((k) => {
      if (props[k]?.type === "object") {
        template[k] = {};
      } else if (props[k]?.type === "array") {
        template[k] = [];
      } else {
        template[k] = required.includes(k) ? "" : "";
      }
    });
    setToolArgsText(JSON.stringify(template, null, 2));
  }, [selectedTool, toolParamsByName]);

  return {
    toolQuery,
    setToolQuery,
    toolResults,
    toolLoading,
    toolError,
    toolList,
    selectedTool,
    setSelectedTool,
    toolArgsText,
    setToolArgsText,
    toolCallResult,
    runToolSearch,
    fetchToolList,
    runToolCall,
  };
};

export default useTools;
