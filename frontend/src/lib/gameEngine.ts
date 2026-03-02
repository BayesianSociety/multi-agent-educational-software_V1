import type { BlockType, Level, RuntimeState } from "@/lib/types";

export interface Instruction {
  opcode: BlockType;
  distance: number;
}

export function compileBlocks(blocks: BlockType[]): Instruction[] {
  return blocks.map((block) => ({
    opcode: block,
    distance: block === "MOVE" ? 1 : 2
  }));
}

export function runInstruction(
  level: Level,
  currentX: number,
  instruction: Instruction
): RuntimeState {
  const nextX = currentX + instruction.distance;
  const hitBlockedTile = level.blockedTiles.includes(nextX);

  if (hitBlockedTile) {
    return {
      x: nextX,
      success: false,
      failed: true,
      reason: "The dog landed in a blocked tile. Try a Jump block."
    };
  }

  if (nextX >= level.goalX) {
    return {
      x: level.goalX,
      success: true,
      failed: false,
      reason: "Success! The dog reached the bone."
    };
  }

  return {
    x: nextX,
    success: false,
    failed: false,
    reason: "Running..."
  };
}

export function evaluateProgram(level: Level, blocks: BlockType[]): RuntimeState {
  const instructions = compileBlocks(blocks);
  let x = level.startX;

  for (const instruction of instructions) {
    const state = runInstruction(level, x, instruction);
    x = state.x;
    if (state.success || state.failed) {
      return state;
    }
  }

  return {
    x,
    success: false,
    failed: true,
    reason: "The dog did not reach the bone yet. Add more blocks."
  };
}
