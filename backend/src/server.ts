import cors from "cors";
import express from "express";
import { PrismaClient } from "@prisma/client";
import { seedLevels } from "./levels";

const prisma = new PrismaClient();
const app = express();
const port = Number(process.env.BACKEND_PORT || 3001);

app.use(cors());
app.use(express.json());

function parseBlockedTiles(value: string): number[] {
  try {
    const parsed = JSON.parse(value) as unknown;
    if (!Array.isArray(parsed)) {
      return [];
    }
    return parsed.filter((tile): tile is number => typeof tile === "number");
  } catch {
    return [];
  }
}

async function seedDatabase(): Promise<void> {
  for (const level of seedLevels) {
    await prisma.level.upsert({
      where: { id: level.id },
      create: {
        id: level.id,
        title: level.title,
        narrative: level.narrative,
        startX: level.startX,
        goalX: level.goalX,
        blockedTiles: JSON.stringify(level.blockedTiles)
      },
      update: {
        title: level.title,
        narrative: level.narrative,
        startX: level.startX,
        goalX: level.goalX,
        blockedTiles: JSON.stringify(level.blockedTiles)
      }
    });
  }

  await prisma.progress.upsert({
    where: { id: 1 },
    create: { id: 1, unlockedLevel: 1 },
    update: {}
  });
}

function toLevelResponse(level: {
  id: number;
  title: string;
  narrative: string;
  startX: number;
  goalX: number;
  blockedTiles: string;
}) {
  return {
    id: level.id,
    title: level.title,
    narrative: level.narrative,
    startX: level.startX,
    goalX: level.goalX,
    blockedTiles: parseBlockedTiles(level.blockedTiles)
  };
}

app.get("/health", (_req, res) => {
  res.status(200).json({ status: "ok" });
});

app.get("/api/levels", async (_req, res) => {
  const levels = await prisma.level.findMany({ orderBy: { id: "asc" } });
  res.status(200).json(levels.map(toLevelResponse));
});

app.get("/api/levels/:id", async (req, res) => {
  const id = Number(req.params.id);
  if (!Number.isInteger(id) || id < 1) {
    res.status(400).json({ error: "Invalid level id" });
    return;
  }

  const level = await prisma.level.findUnique({ where: { id } });
  if (!level) {
    res.status(404).json({ error: "Level not found" });
    return;
  }

  res.status(200).json(toLevelResponse(level));
});

app.get("/api/progress", async (_req, res) => {
  const progress = await prisma.progress.findUnique({ where: { id: 1 } });
  res.status(200).json({ unlockedLevel: progress?.unlockedLevel ?? 1 });
});

app.put("/api/progress", async (req, res) => {
  const unlockedLevel = Number(req.body?.unlockedLevel);
  if (!Number.isInteger(unlockedLevel) || unlockedLevel < 1 || unlockedLevel > seedLevels.length) {
    res.status(400).json({ error: "Invalid unlockedLevel" });
    return;
  }

  const current = await prisma.progress.findUnique({ where: { id: 1 } });
  const nextUnlocked = Math.max(current?.unlockedLevel ?? 1, unlockedLevel);

  const updated = await prisma.progress.upsert({
    where: { id: 1 },
    create: { id: 1, unlockedLevel: nextUnlocked },
    update: { unlockedLevel: nextUnlocked }
  });

  res.status(200).json({ unlockedLevel: updated.unlockedLevel });
});

async function startServer(): Promise<void> {
  await seedDatabase();
  app.listen(port, () => {
    // Deterministic startup log line used for local debugging.
    console.log(`Backend listening on ${port}`);
  });
}

startServer().catch(async (error: unknown) => {
  console.error("Failed to start backend", error);
  await prisma.$disconnect();
  process.exit(1);
});

process.on("SIGINT", async () => {
  await prisma.$disconnect();
  process.exit(0);
});

process.on("SIGTERM", async () => {
  await prisma.$disconnect();
  process.exit(0);
});
