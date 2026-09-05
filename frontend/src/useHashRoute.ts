// useHashRoute.ts
// ================
// V16 Phase D §D5 -- fixes D6 (no URL routing; view and selected episode
// lived only in React state, so refresh lost everything and nothing was
// linkable). Hand-rolled hash router, not a library: this app has exactly
// six fixed tabs plus one optional episode-id param, which is much less
// machinery than a real router package provides for. `#/replay/ep-0042`
// survives a refresh and is shareable, per the spec.
import { useCallback, useEffect, useState } from "react";

export type Tab = "home" | "validation" | "baseline" | "training" | "analysis" | "replay";

const VALID_TABS: Tab[] = ["home", "validation", "baseline", "training", "analysis", "replay"];

interface RouteState {
  tab: Tab;
  episodeId: string | null;
}

function parseHash(): RouteState {
  const raw = window.location.hash.replace(/^#\/?/, ""); // strip a leading "#" or "#/"
  const [tabPart, episodePart] = raw.split("/");
  const tab = (VALID_TABS as string[]).includes(tabPart) ? (tabPart as Tab) : "home";
  const episodeId = tab === "replay" && episodePart ? decodeURIComponent(episodePart) : null;
  return { tab, episodeId };
}

export function useHashRoute() {
  const [route, setRoute] = useState<RouteState>(() => parseHash());

  useEffect(() => {
    function onHashChange() {
      setRoute(parseHash());
    }
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  const navigate = useCallback((tab: Tab, episodeId?: string | null) => {
    const nextHash = tab === "replay" && episodeId ? `#/replay/${encodeURIComponent(episodeId)}` : `#/${tab}`;
    if (window.location.hash === nextHash) {
      // Same hash (e.g. re-clicking the already-active tab) -- "hashchange"
      // won't fire on its own, so sync state directly instead.
      setRoute({ tab, episodeId: episodeId ?? null });
    } else {
      window.location.hash = nextHash;
    }
  }, []);

  return { tab: route.tab, episodeId: route.episodeId, navigate };
}
