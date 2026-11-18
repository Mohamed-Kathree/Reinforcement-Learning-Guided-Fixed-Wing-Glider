// useHashRoute.ts
// ================
// Hand-rolled hash router: fixed tabs plus optional episode ids for Flight
// Replay. `#/replay/<id>` opens one flight; `#/replay/<idA>/<idB>` opens two
// side by side (used for paired benchmark flights). Survives refresh and is
// shareable.
import { useCallback, useEffect, useState } from "react";

export type Tab = "home" | "validation" | "research" | "replay" | "baseline" | "training" | "analysis" | "flighttest"
  | "groundstation" | "flightlogs";

const VALID_TABS: Tab[] = ["home", "validation", "research", "replay", "baseline", "training", "analysis", "flighttest",
  "groundstation", "flightlogs"];

interface RouteState {
  tab: Tab;
  episodeId: string | null;
  episodeIdB: string | null;
}

function parseHash(): RouteState {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [tabPart, a, b] = raw.split("/");
  const tab = (VALID_TABS as string[]).includes(tabPart) ? (tabPart as Tab) : "home";
  const isReplay = tab === "replay";
  return {
    tab,
    episodeId: isReplay && a ? decodeURIComponent(a) : null,
    episodeIdB: isReplay && b ? decodeURIComponent(b) : null,
  };
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

  const navigate = useCallback((tab: Tab, episodeId?: string | null, episodeIdB?: string | null) => {
    let nextHash = `#/${tab}`;
    if (tab === "replay" && episodeId) {
      nextHash += `/${encodeURIComponent(episodeId)}`;
      if (episodeIdB) nextHash += `/${encodeURIComponent(episodeIdB)}`;
    }
    if (window.location.hash === nextHash) {
      setRoute({ tab, episodeId: episodeId ?? null, episodeIdB: episodeIdB ?? null });
    } else {
      window.location.hash = nextHash;
    }
    window.scrollTo({ top: 0 });
  }, []);

  return { tab: route.tab, episodeId: route.episodeId, episodeIdB: route.episodeIdB, navigate };
}
