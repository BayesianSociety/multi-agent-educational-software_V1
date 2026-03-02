"use client";

import { useEffect, useMemo, useState } from "react";
import { fetchLevels } from "@/api";
import { compileBlocks, runInstruction } from "@/lib/gameEngine";
import type { BlockType, Level } from "@/lib/types";

const STORAGE_KEY = "dog-bone-unlocked-level";

export default function HomePage() {
  const [levels, setLevels] = useState<Level[]>([]);
  const [selectedLevelId, setSelectedLevelId] = useState(1);
  const [workspace, setWorkspace] = useState<BlockType[]>([]);
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState<number | null>(null);
  const [dogX, setDogX] = useState(0);
  const [statusText, setStatusText] = useState("Build a program, then press Run.");
  const [unlockedLevel, setUnlockedLevel] = useState(1);

  useEffect(() => {
    const saved = Number(globalThis.localStorage?.getItem(STORAGE_KEY) || "1");
    setUnlockedLevel(Number.isFinite(saved) && saved > 0 ? saved : 1);

    fetchLevels("")
      .then((data) => {
        setLevels(data);
      })
      .catch(() => {
        setStatusText("Could not load levels.");
      });
  }, []);

  const level = useMemo(() => levels.find((item) => item.id === selectedLevelId) ?? levels[0], [levels, selectedLevelId]);
  const plannedActions = useMemo(() => {
    if (!level || workspace.length === 0) {
      return [];
    }
    let x = level.startX;
    return compileBlocks(workspace).map((instruction, index) => {
      x = Math.min(level.goalX, x + instruction.distance);
      return {
        index,
        opcode: instruction.opcode,
        x
      };
    });
  }, [level, workspace]);

  useEffect(() => {
    if (level) {
      setDogX(level.startX);
      setWorkspace([]);
      setCurrentStep(null);
      setStatusText("Build a program, then press Run.");
    }
  }, [level?.id]);

  function pushBlock(block: BlockType) {
    if (isRunning) {
      return;
    }
    setWorkspace((prev) => [...prev, block]);
  }

  function removeBlock(index: number) {
    if (isRunning) {
      return;
    }
    setWorkspace((prev) => prev.filter((_, i) => i !== index));
  }

  function resetRun() {
    if (!level) {
      return;
    }
    setIsRunning(false);
    setCurrentStep(null);
    setDogX(level.startX);
    setStatusText("Program reset. Edit blocks and run again.");
  }

  function runProgram() {
    if (!level || workspace.length === 0 || isRunning) {
      return;
    }

    const instructions = compileBlocks(workspace);
    let x = level.startX;
    let stepIndex = 0;

    setIsRunning(true);
    setDogX(level.startX);
    setStatusText("Running deterministic sequence...");

    const timer = globalThis.setInterval(() => {
      const instruction = instructions[stepIndex];
      if (!instruction) {
        globalThis.clearInterval(timer);
        setIsRunning(false);
        setCurrentStep(null);
        setStatusText("The dog did not reach the bone yet. Add more blocks.");
        return;
      }

      setCurrentStep(stepIndex);
      const state = runInstruction(level, x, instruction);
      x = state.x;
      setDogX(state.x);

      if (state.failed || state.success) {
        globalThis.clearInterval(timer);
        setIsRunning(false);
        setCurrentStep(null);
        setStatusText(state.reason);

        if (state.success) {
          const nextUnlocked = Math.min(10, Math.max(unlockedLevel, level.id + 1));
          setUnlockedLevel(nextUnlocked);
          globalThis.localStorage?.setItem(STORAGE_KEY, String(nextUnlocked));
        }
        return;
      }

      stepIndex += 1;
    }, 650);
  }

  if (!level) {
    return <main className="shell">Loading levels...</main>;
  }

  return (
    <main className="shell">
      <h1>Dog Bone Blocks</h1>
      <p className="subtitle">Guide the dog to the bone by sequencing Move and Jump blocks.</p>

      <section className="panel" aria-label="Level picker">
        <h2>Levels</h2>
        <div className="levelGrid">
          {levels.map((item) => {
            const locked = item.id > unlockedLevel;
            return (
              <button
                key={item.id}
                type="button"
                className={`levelButton ${item.id === level.id ? "active" : ""}`}
                onClick={() => setSelectedLevelId(item.id)}
                disabled={locked || isRunning}
                aria-label={locked ? `Level ${item.id} locked` : `Open level ${item.id}`}
              >
                L{item.id} {locked ? "(Locked)" : ""}
              </button>
            );
          })}
        </div>
        <p className="narration" aria-live="polite">
          <strong>{level.title}:</strong> {level.narrative}
        </p>
      </section>

      <section className="panel" aria-label="Block palette">
        <h2>Blocks</h2>
        <div className="actions">
          <button type="button" onClick={() => pushBlock("MOVE")} disabled={isRunning}>
            Add Move (1)
          </button>
          <button type="button" onClick={() => pushBlock("JUMP")} disabled={isRunning}>
            Add Jump
          </button>
          <button type="button" onClick={runProgram} disabled={isRunning || workspace.length === 0}>
            Run / Start
          </button>
          <button type="button" onClick={resetRun}>
            Reset
          </button>
        </div>
      </section>

      <section className="panel" aria-label="Program workspace">
        <h2>Workspace</h2>
        <p className="smallHelp">Keyboard: Tab to a block, then Enter/Space to remove it.</p>
        <ol className="workspaceList">
          {workspace.map((block, index) => (
            <li key={`${block}-${index}`}>
              <button
                type="button"
                onClick={() => removeBlock(index)}
                disabled={isRunning}
                className={currentStep === index ? "current" : ""}
                aria-current={currentStep === index ? "step" : undefined}
                aria-label={`Step ${index + 1}: ${block}. Press to remove.`}
              >
                Step {index + 1}: {block} {currentStep === index ? "(Current step)" : ""}
              </button>
            </li>
          ))}
        </ol>
      </section>

      <section className="panel" aria-label="Playfield">
        <h2>Playfield</h2>
        <p className="status" aria-live="polite">
          {statusText}
        </p>
        <div className="gameBoard" role="img" aria-label="Dog path and bone goal over level background">
          <div className="dog token" style={{ left: `${(dogX / level.goalX) * 100}%` }}>
            Dog
          </div>
          <div className="bone token">Bone</div>
          {level.blockedTiles.map((blocked) => (
            <div
              key={blocked}
              className="blocked token"
              style={{ left: `${(blocked / level.goalX) * 100}%` }}
              aria-label={`Blocked tile at ${blocked}`}
              title={`Blocked tile ${blocked}`}
            >
              X
            </div>
          ))}
          {plannedActions.map((action) => (
            <div
              key={`${action.opcode}-${action.index}`}
              className={`actionMarker ${currentStep === action.index ? "active" : ""}`}
              style={{ left: `${(action.x / level.goalX) * 100}%` }}
              aria-label={`Planned action ${action.index + 1}: ${action.opcode}`}
              title={`Step ${action.index + 1}: ${action.opcode}`}
            >
              {action.opcode === "MOVE" ? "M" : "J"}
            </div>
          ))}
        </div>
      </section>
    </main>
  );
}
