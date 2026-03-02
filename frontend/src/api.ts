import type { Level } from "@/lib/types";

const localLevels: Level[] = [
  { id: 1, title: "First Steps", narrative: "Use Move to walk toward the bone.", startX: 0, goalX: 4, blockedTiles: [] },
  { id: 2, title: "Leap Time", narrative: "A hole appears. Use Jump to cross it.", startX: 0, goalX: 6, blockedTiles: [3] },
  { id: 3, title: "Two Jumps", narrative: "Plan two jumps so the dog stays safe.", startX: 0, goalX: 8, blockedTiles: [2, 5] },
  { id: 4, title: "Steady Rhythm", narrative: "Alternate Move and Jump carefully.", startX: 0, goalX: 9, blockedTiles: [4] },
  { id: 5, title: "Tiny Maze", narrative: "Watch each step and avoid blocked spots.", startX: 0, goalX: 10, blockedTiles: [3, 7] },
  { id: 6, title: "Skip and Sprint", narrative: "Build a longer sequence to reach the bone.", startX: 0, goalX: 11, blockedTiles: [2, 6, 9] },
  { id: 7, title: "Careful Crossing", narrative: "Choose jumps only where they are needed.", startX: 0, goalX: 12, blockedTiles: [5, 8] },
  { id: 8, title: "Debug Path", narrative: "Run, inspect, then fix your block plan.", startX: 0, goalX: 13, blockedTiles: [4, 10] },
  { id: 9, title: "Long Trail", narrative: "Keep a deterministic order to finish.", startX: 0, goalX: 14, blockedTiles: [3, 7, 11] },
  { id: 10, title: "Bone Master", narrative: "Final challenge: plan and execute perfectly.", startX: 0, goalX: 15, blockedTiles: [4, 9, 12] }
];

export async function fetchLevels(baseUrl = ""): Promise<Level[]> {
  try {
    const response = await fetch(`${baseUrl}/api/levels`, { method: "GET", cache: "no-store" });
    if (!response.ok) {
      throw new Error("Failed to load levels");
    }

    const data = (await response.json()) as Level[];
    return data.length > 0 ? data : localLevels;
  } catch {
    return localLevels;
  }
}

export async function fetchLevelById(id: number, baseUrl = ""): Promise<Level> {
  try {
    const response = await fetch(`${baseUrl}/api/levels/${id}`, { method: "GET", cache: "no-store" });
    if (!response.ok) {
      throw new Error("Failed to load level");
    }
    return (await response.json()) as Level;
  } catch {
    const fallback = localLevels.find((level) => level.id === id);
    if (!fallback) {
      throw new Error(`Level ${id} not found`);
    }
    return fallback;
  }
}

export { localLevels };
