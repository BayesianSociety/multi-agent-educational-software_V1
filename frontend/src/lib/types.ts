export type BlockType = "MOVE" | "JUMP";

export interface Level {
  id: number;
  title: string;
  narrative: string;
  startX: number;
  goalX: number;
  blockedTiles: number[];
  minimumUnlockedLevel?: number;
}

export interface RuntimeState {
  x: number;
  success: boolean;
  failed: boolean;
  reason: string;
}

export interface Progress {
  unlockedLevel: number;
}
