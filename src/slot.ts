/**
 * Hot reload without mixing releases: a slot holds one ready router. reload() builds and validates a
 * complete replacement first, swaps the pointer atomically, then closes (drains) the old router, so
 * in-flight requests finish on the snapshot they started with and every result names exactly one
 * release. A failed reload keeps the previous healthy router. Rollback is a reload of an earlier
 * pinned release.
 */
import type { RouteRequest, RouteResult, Router } from './router.ts';

export type ReloadOutcome = { ok: true; releaseId: string; previousReleaseId: string } | { ok: false; releaseId: string; error: unknown };

export class RouterSlot {
  #current: Router;
  #reloading: Promise<ReloadOutcome> | null = null;
  #closing: Promise<void> | null = null;
  /** Routers retired by reloads that are still draining; close() waits for them too. */
  readonly #retiring = new Set<Promise<void>>();

  constructor(initial: Router) {
    this.#current = initial;
  }

  get current(): Router {
    return this.#current;
  }

  /** Waits for the current router's background work (monitoring, observer delivery); retired routers drain on close. */
  flush(): Promise<void> {
    return this.#current.flush();
  }

  /** After close() has begun, the current router answers CLOSED: no request is actionable. */
  route(request: RouteRequest): Promise<RouteResult> {
    return this.#current.route(request);
  }

  /**
   * Loads a replacement (e.g. () => loadRouter({...})) and swaps only if it loads. One reload at a time.
   * Refused once close() has begun; a replacement that finishes loading after that is closed, not installed.
   */
  reload(load: () => Promise<Router>): Promise<ReloadOutcome> {
    if (this.#closing) return Promise.resolve({ ok: false, releaseId: this.#current.releaseId, error: new Error('the router slot is closed') });
    if (this.#reloading) return this.#reloading.then(() => this.reload(load));
    this.#reloading = (async (): Promise<ReloadOutcome> => {
      const previous = this.#current;
      let next: Router;
      try {
        next = await load();
      } catch (error) {
        return { ok: false, releaseId: previous.releaseId, error };
      }
      if (this.#closing) {
        await next.close();
        return { ok: false, releaseId: previous.releaseId, error: new Error('the router slot closed while the replacement was loading') };
      }
      this.#current = next;
      const draining = previous.close();
      this.#retiring.add(draining);
      await draining;
      this.#retiring.delete(draining);
      return { ok: true, releaseId: next.releaseId, previousReleaseId: previous.releaseId };
    })().finally(() => { this.#reloading = null; });
    return this.#reloading;
  }

  /** Terminal and idempotent: closes the current router, waits for any pending reload and every retiring router to drain. */
  close(): Promise<void> {
    this.#closing ??= (async () => {
      const current = this.#current.close();
      await this.#reloading?.catch(() => undefined);
      await Promise.all([current, this.#current.close(), ...this.#retiring]);
    })();
    return this.#closing;
  }
}
