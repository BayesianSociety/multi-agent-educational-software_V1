"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const cors_1 = __importDefault(require("cors"));
const express_1 = __importDefault(require("express"));
const client_1 = require("@prisma/client");
const levels_1 = require("./levels");
const prisma = new client_1.PrismaClient();
const app = (0, express_1.default)();
const port = Number(process.env.BACKEND_PORT || 3001);
app.use((0, cors_1.default)());
app.use(express_1.default.json());
function parseBlockedTiles(value) {
    try {
        const parsed = JSON.parse(value);
        if (!Array.isArray(parsed)) {
            return [];
        }
        return parsed.filter((tile) => typeof tile === "number");
    }
    catch {
        return [];
    }
}
async function seedDatabase() {
    for (const level of levels_1.seedLevels) {
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
function toLevelResponse(level) {
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
    if (!Number.isInteger(unlockedLevel) || unlockedLevel < 1 || unlockedLevel > levels_1.seedLevels.length) {
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
async function startServer() {
    await seedDatabase();
    app.listen(port, () => {
        // Deterministic startup log line used for local debugging.
        console.log(`Backend listening on ${port}`);
    });
}
startServer().catch(async (error) => {
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
