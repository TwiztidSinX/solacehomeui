  import React, { useEffect, useRef } from 'react';
  import * as d3 from 'd3';

  // Enhanced Node interface to track physics and glow state
  interface Node extends d3.SimulationNodeDatum {
    id: string;
    label: string;
    group: string;
    value: number;
    glowUntil?: number;
    lastHit?: number;
    glowColor?: string;
  }

  interface Link {
    source: string;
    target: string;
    value: number;
  }

  interface GraphData {
    nodes: Node[];
    edges: { from: string; to: string; value: number }[];
  }

  interface MemoryGraphProps {
    data: GraphData | null;
  }

  const MemoryGraph: React.FC<MemoryGraphProps> = ({ data }) => {
    const svgRef = useRef<SVGSVGElement>(null);

    useEffect(() => {
      if (!data || !svgRef.current) return;

      const svg = d3.select(svgRef.current);
      svg.selectAll("*").remove();

      const width = svg.node()!.getBoundingClientRect().width;
      const height = svg.node()!.getBoundingClientRect().height;

      if (!data.nodes || !data.edges || data.nodes.length === 0) {
        svg.append("text")
          .attr("x", "50%")
          .attr("y", "50%")
          .attr("text-anchor", "middle")
          .attr("fill", "white")
          .text("No graph data to display.");
        return;
      }

      const links: Link[] = data.edges.map(edge => ({ source: edge.from, target: edge.to, value: edge.value
  }));
      const nodes: Node[] = data.nodes.map(n => ({ ...n }));

      const zoomGroup = svg.append("g");
      svg.call(d3.zoom<SVGSVGElement, unknown>().on("zoom", (event) => {
        zoomGroup.attr("transform", event.transform.toString());
      }));

      const simulation = d3.forceSimulation(nodes)
        .force("link", d3.forceLink(links).id((d: any) => d.id).distance(150).strength(0.1))
        .force("charge", d3.forceManyBody().strength(-400)) // Repulsion force for initial layout
        .force("collision", d3.forceCollide().radius(25).strength(0.9)) // Bounciness
        .force("center", d3.forceCenter(width / 2, height / 2))
        .velocityDecay(0.15) // KEY CHANGE: Low friction for gliding
        .alphaMin(0.001);

      const valueExtent = d3.extent(nodes, d => d.value || 0) as [number, number];
      const colorScale = d3.scaleSequential(d3.interpolateViridis)
        .domain(valueExtent[0] < valueExtent[1] ? valueExtent : [0, 1]);

      const baseNodeColor = "#4b5563";
      const collisionColors = ["#ff00ff", "#00ffff", "#ff9900", "#00ff00", "#ff0000", "#ffff00"];

      const link = zoomGroup.append("g")
        .selectAll("line")
        .data(links)
        .enter().append("line")
        .attr("stroke-width", d => Math.sqrt(d.value || 1))
        .attr("stroke", "#0026ffff")
        .attr("stroke-opacity", 0.6);

      const nodeElements = zoomGroup.append("g")
        .selectAll<SVGCircleElement, Node>("circle")
        .data(nodes)
        .enter().append("circle")
        .attr("r", 12)
        .attr("fill", d => d.value > 0 ? colorScale(d.value) : baseNodeColor)
        .attr("stroke", "#7700ffff")
        .attr("stroke-width", 2.5);

      const drag = d3.drag<SVGCircleElement, Node>()
        .on("start", (event: any, d: Node) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
          (d as any)._dragStart = Date.now();
          (d as any)._lastPos = { x: event.x, y: event.y };
          (d as any)._velocity = { x: 0, y: 0 };
        })
        .on("drag", (event: any, d: Node) => {
          d.fx = event.x;
          d.fy = event.y;

          const last = (d as any)._lastPos;
          (d as any)._velocity = {
            x: event.x - last.x,
            y: event.y - last.y
          };
          (d as any)._lastPos = { x: event.x, y: event.y };

          // charge factor grows after 500ms hold
          const holdTime = Date.now() - (d as any)._dragStart;
          if (holdTime > 500) {
            const factor = Math.min(1, (holdTime - 500) / 1500); // max at ~2s
            d.glowColor = "#ffcc00";
            d.glowUntil = Date.now() + 100; // constantly refreshed
            (d as any)._chargeFactor = factor;
          } else {
            (d as any)._chargeFactor = 0;
          }
        })
        .on("end", (event: any, d: Node) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;

          const holdTime = Date.now() - (d as any)._dragStart;
          const velocity = (d as any)._velocity || { x: 0, y: 0 };
          const chargeFactor = (d as any)._chargeFactor || 0;

          if (holdTime < 500 || chargeFactor === 0) {
            // short drag → no flick
            d.vx = 0;
            d.vy = 0;
          } else {
            // charged flick → apply velocity
            d.vx = velocity.x * 5 * chargeFactor;
            d.vy = velocity.y * 5 * chargeFactor;
          }

          d.glowColor = undefined;
          d.glowUntil = undefined;

          simulation.alpha(1).restart();
        });

      nodeElements.call(drag);

      nodeElements.append("title")
        .text(d => `${d.label}\nType: ${d.group}\nRecalls: ${d.value || 0}`); // SYNTAX FIX

      const label = zoomGroup.append("g")
        .selectAll("text")
        .data(nodes)
        .enter().append("text")
        .text(d => d.label)
        .attr("fill", "white")
        .style("font-size", "12px")
        .style("text-anchor", "middle")
        .style("pointer-events", "none");

      setTimeout(() => {
        simulation.stop();
        simulation.force("charge", null);
        simulation.force("center", null);
      }, 3000);

      const glowTimer = d3.timer(() => {
        const now = Date.now();
        nodeElements.each(function(d) {
          const circle = d3.select(this);
          if (d.glowUntil && now < d.glowUntil) {
            const remaining = d.glowUntil - now;
            const progress = remaining / 10000;
            const easedProgress = d3.easeCubicOut(progress);

            const currentColor = d3.interpolateRgb(
              d.value > 0 ? colorScale(d.value) : baseNodeColor,
              d.glowColor || baseNodeColor
            )(easedProgress);

            circle.attr("fill", currentColor as string);

          } else if (d.glowUntil && now >= d.glowUntil) {
            d.glowUntil = undefined;
            d.glowColor = undefined;
            circle.transition().duration(500).attr("fill", d.value > 0 ? colorScale(d.value) :
  baseNodeColor);
          }
        });
      });

      simulation.on("tick", () => {
        const q = d3.quadtree<Node>().x(d => d.x!).y(d => d.y!).addAll(nodes);
        for (const n1 of nodes) {
          q.visit((quad) => {
            const n2 = (quad as d3.QuadtreeLeaf<Node>).data;
            if (n2 && n2 !== n1) {
              const dx = n1.x! - n2.x!;
              const dy = n1.y! - n2.y!;
              const distance = Math.sqrt(dx * dx + dy * dy); // SYNTAX FIX
              const minDistance = 24;

              if (distance < minDistance) {
                const now = Date.now();
                [n1, n2].forEach(n => {
                  if (!n.lastHit || now - n.lastHit > 30000) {
                    n.glowColor = collisionColors[Math.floor(Math.random() * collisionColors.length)];
                  }
                  n.lastHit = now;
                  n.glowUntil = now + 10000;
                });
              }
            }
            return false;
          });
        }

        link
          .attr("x1", d => (d.source as any).x)
          .attr("y1", d => (d.source as any).y)
          .attr("x2", d => (d.target as any).x)
          .attr("y2", d => (d.target as any).y);

        nodeElements
          .attr("cx", d => d.x!)
          .attr("cy", d => d.y!);

        label
          .attr("x", d => d.x!)
          .attr("y", d => d.y! + 25);
      });

      return () => {
        simulation.stop();
        glowTimer.stop();
      };

    }, [data]);

    return (
      <div className="tab-content flex-1 p-4 w-full">
        <svg ref={svgRef} className="w-full h-full"></svg>
      </div>
    );
  };

  export default MemoryGraph;