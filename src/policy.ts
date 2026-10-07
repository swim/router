/**
 * The routing policy: declarative routes (one classifier head each in v1), their priority, suppression
 * and what to do on abstention. Destinations are opaque application identifiers; the router never
 * interprets or executes them. Thresholds are NOT part of the policy: they come only from the
 * validated classifier artifact. Changing priority or suppression is a new policy digest and needs a
 * fresh final-system evaluation.
 */
import { RouterLoadError } from './errors.ts';

export const POLICY_SCHEMA = 'liquidau-router-policy/1';

export interface RouteSpec {
  /** The classifier head this route acts on. */
  id: string;
  description?: string;
  /** Opaque, application-owned. */
  destination: string;
  /** Authoring/training seeds; never evaluation evidence. */
  examples?: string[];
}

export interface RouterPolicy {
  schema: typeof POLICY_SCHEMA;
  routes: RouteSpec[];
  /** Every route exactly once, first wins. */
  priority: string[];
  /** While any `when` head is at/above its review floor (or rule-fired), none of `heads` may fire. */
  suppress: Array<{ when: string[]; heads: string[] }>;
  /** 'review': send abstentions to a human; 'none': abstain. */
  onAbstain: 'none' | 'review';
}

const isObject = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const names = (v: unknown): v is string[] => Array.isArray(v) && v.length > 0 && v.every((s) => typeof s === 'string' && s.length > 0);

/**
 * Problems with a policy (empty when valid). With `heads` (the classifier's), the routes must be
 * exactly those heads - extra classifier heads are refused in v1, since guard heads are not part of
 * this schema. With `ruleLabels`, every rule label must be a route.
 */
export function policyProblems(raw: unknown, context: { heads?: readonly string[]; ruleLabels?: readonly string[] } = {}): string[] {
  if (!isObject(raw)) return ['the policy is not an object'];
  if (raw.schema !== POLICY_SCHEMA) return [`unsupported policy schema ${JSON.stringify(raw.schema)} (this router reads '${POLICY_SCHEMA}')`];
  const p: string[] = [];
  for (const k of Object.keys(raw)) if (!['schema', 'routes', 'priority', 'suppress', 'onAbstain'].includes(k)) p.push(`unknown field ${k}`);
  const ids: string[] = [];
  if (!Array.isArray(raw.routes) || !raw.routes.length) p.push('routes must be a non-empty array');
  else raw.routes.forEach((r: unknown, i: number) => {
    if (!isObject(r)) { p.push(`routes[${i}] is not an object`); return; }
    for (const k of Object.keys(r)) if (!['id', 'description', 'destination', 'examples'].includes(k)) p.push(`routes[${i}] has unknown field ${k}`);
    if (typeof r.id !== 'string' || !r.id.length) p.push(`routes[${i}].id must be a non-empty string`);
    else if (ids.includes(r.id)) p.push(`route id ${r.id} is duplicated`);
    else ids.push(r.id);
    if (typeof r.destination !== 'string' || !r.destination.length) p.push(`routes[${i}].destination must be a non-empty string`);
    if (r.description !== undefined && typeof r.description !== 'string') p.push(`routes[${i}].description must be a string`);
    if (r.examples !== undefined && !(Array.isArray(r.examples) && r.examples.every((e) => typeof e === 'string'))) p.push(`routes[${i}].examples must be strings`);
  });
  const known = new Set(ids);
  if (!names(raw.priority)) p.push('priority must be a non-empty array of route ids');
  else {
    const seen = new Set<string>();
    for (const h of raw.priority) {
      if (seen.has(h)) p.push(`priority lists ${h} twice`);
      seen.add(h);
      if (!known.has(h)) p.push(`priority names ${h}, which is not a route`);
    }
    for (const id of ids) if (!seen.has(id)) p.push(`route ${id} is missing from priority`);
  }
  if (!Array.isArray(raw.suppress)) p.push('suppress must be an array (empty for none)');
  else {
    const edges = new Map<string, Set<string>>();
    raw.suppress.forEach((s: unknown, i: number) => {
      if (!isObject(s) || !names(s.when) || !names(s.heads) || Object.keys(s).some((k) => k !== 'when' && k !== 'heads')) { p.push(`suppress[${i}] must be { when: string[], heads: string[] }`); return; }
      for (const h of [...s.when, ...s.heads]) if (!known.has(h)) p.push(`suppress[${i}] names ${h}, which is not a route`);
      for (const h of s.heads) if (s.when.includes(h)) p.push(`suppress[${i}]: ${h} suppresses itself`);
      for (const w of s.when) for (const h of s.heads) if (w !== h) { if (!edges.has(w)) edges.set(w, new Set()); edges.get(w)!.add(h); }
    });
    const cycle = findCycle(edges);
    if (cycle) p.push(`suppression cycle ${cycle.join(' -> ')}: v1 refuses mutual suppression; split the routes or drop one direction and re-evaluate the release`);
  }
  if (raw.onAbstain !== 'none' && raw.onAbstain !== 'review') p.push("onAbstain must be 'none' or 'review'");
  if (context.heads) {
    const heads = new Set(context.heads);
    for (const id of ids) if (!heads.has(id)) p.push(`route ${id} has no classifier head`);
    for (const h of heads) if (!known.has(h)) p.push(`classifier head ${h} is not a route (v1 refuses undeclared extra heads)`);
  }
  for (const label of new Set(context.ruleLabels ?? [])) if (!known.has(label)) p.push(`rule label ${label} is not a route`);
  return p;
}

/** Throws RouterLoadError POLICY_INVALID with every problem found. */
export function validatePolicy(raw: unknown, context: { heads?: readonly string[]; ruleLabels?: readonly string[] } = {}): RouterPolicy {
  const p = policyProblems(raw, context);
  if (p.length) throw new RouterLoadError('POLICY_INVALID', 'the policy is invalid', p);
  return raw as RouterPolicy;
}

function findCycle(edges: Map<string, Set<string>>): string[] | null {
  const state = new Map<string, 'visiting' | 'done'>();
  const path: string[] = [];
  const visit = (n: string): string[] | null => {
    if (state.get(n) === 'visiting') return [...path.slice(path.indexOf(n)), n];
    if (state.get(n) === 'done') return null;
    state.set(n, 'visiting');
    path.push(n);
    for (const m of edges.get(n) ?? []) { const c = visit(m); if (c) return c; }
    path.pop();
    state.set(n, 'done');
    return null;
  };
  for (const n of [...edges.keys()].sort()) { const c = visit(n); if (c) return c; }
  return null;
}
