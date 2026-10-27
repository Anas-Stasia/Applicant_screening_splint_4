/**
 * Applicant Screening API — JavaScript client for the Frontend squad.
 *
 * Works in the browser (React / Vue / Vite) and in Node 18+ (built-in fetch).
 * No dependencies. Copy this file into your app, or generate full TypeScript
 * types from the live spec instead:
 *
 *     npx openapi-typescript http://localhost:8000/openapi.json -o src/api/screening.d.ts
 *
 * Usage:
 *     import { ScreeningClient, ScreeningApiError } from "./screening_client.js";
 *     const api = new ScreeningClient({ baseUrl: "http://localhost:8000" });
 *     const { result } = await api.match(resumeText);
 *     if (result.status === "low_confidence") showManualReviewBadge();
 *
 * Every non-2xx response throws ScreeningApiError with the server's stable
 * `code` (e.g. "TEXT_TOO_SHORT", "BATCH_TOO_LARGE") so the UI can branch on it.
 */

export class ScreeningApiError extends Error {
  /**
   * @param {number} status   HTTP status
   * @param {{code:string,message:string,details?:any,request_id?:string}} error
   */
  constructor(status, error) {
    super(error.message);
    this.name = "ScreeningApiError";
    this.status = status;
    this.code = error.code;
    this.details = error.details;
    this.requestId = error.request_id;
  }
}

/** Friendly UI copy for the error codes users can actually fix. */
export const ERROR_MESSAGES = {
  EMPTY_TEXT: "Please paste the resume text.",
  TEXT_TOO_SHORT: "That resume is too short to score reliably.",
  TEXT_TOO_LONG: "That resume is too long. Please trim it.",
  NO_TEXT_CONTENT: "We couldn't find any words in that text.",
  UNKNOWN_CATEGORY: "That job category doesn't exist.",
  BATCH_TOO_LARGE: "Too many resumes at once.",
  MODEL_NOT_READY: "The screening service is starting up. Try again in a moment.",
};

export class ScreeningClient {
  /**
   * @param {{baseUrl?: string, apiKey?: string, timeoutMs?: number, maxBatchSize?: number}} [opts]
   */
  constructor({ baseUrl = "http://localhost:8000", apiKey, timeoutMs = 30000, maxBatchSize = 64 } = {}) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
    this.maxBatchSize = maxBatchSize;
  }

  async #request(method, path, body, { retries = 2 } = {}) {
    const headers = { "Content-Type": "application/json" };
    if (this.apiKey) headers["X-API-Key"] = this.apiKey;

    for (let attempt = 0; ; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.timeoutMs);
      let res;
      try {
        res = await fetch(this.baseUrl + path, {
          method,
          headers,
          body: body === undefined ? undefined : JSON.stringify(body),
          signal: controller.signal,
        });
      } finally {
        clearTimeout(timer);
      }
      if (res.ok) return res.json();

      const payload = await res.json().catch(() => ({}));
      const error = payload.error ?? { code: "HTTP_ERROR", message: res.statusText };
      // 503 = model still loading: worth a short retry. Anything else is final.
      if (res.status === 503 && attempt < retries) {
        await new Promise((r) => setTimeout(r, 1000 * (attempt + 1)));
        continue;
      }
      throw new ScreeningApiError(res.status, error);
    }
  }

  health() { return this.#request("GET", "/health"); }
  categories() { return this.#request("GET", "/v1/categories"); }
  modelInfo() { return this.#request("GET", "/v1/model"); }

  /** Classify one resume. */
  match(text, topK = 3) {
    return this.#request("POST", "/v1/match", { text, top_k: topK });
  }

  /**
   * Classify many resumes. Splits into server-sized chunks automatically and
   * returns results in the original order.
   * @param {{id?: string, text: string}[]} items
   */
  async matchBatch(items, topK = 3) {
    const results = [];
    for (let i = 0; i < items.length; i += this.maxBatchSize) {
      const chunk = items.slice(i, i + this.maxBatchSize).map((it, j) => ({ id: it.id ?? String(i + j), text: it.text }));
      const res = await this.#request("POST", "/v1/match/batch", { items: chunk, top_k: topK });
      results.push(...res.results);
    }
    return results;
  }

  /**
   * Rank applicants for one job.
   * @param {{category?: string, description?: string}} job  exactly one of the two
   * @param {{id?: string, text: string}[]} candidates
   */
  screen(job, candidates, topK) {
    return this.#request("POST", "/v1/screen", { job, candidates, ...(topK ? { top_k: topK } : {}) });
  }

  /** Rank the indexed resume cohort against a job. */
  rankCohort(job, topK = 10) {
    return this.#request("POST", "/v1/cohort/rank", { job, top_k: topK });
  }
}
