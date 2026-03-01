export async function fetchLevels(baseUrl: string): Promise<unknown> {
  const response = await fetch(`${baseUrl}/api/levels`, { method: "GET" });
  if (!response.ok) {
    throw new Error("Failed to load levels");
  }
  return response.json();
}
