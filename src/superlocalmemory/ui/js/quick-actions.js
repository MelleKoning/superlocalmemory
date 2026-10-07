// SuperLocalMemory v3.4.1 — Quick Insight Actions
// Copyright (c) 2026 Varun Pratap Bhardwaj — AGPL-3.0-or-later
// 5 one-click intelligence buttons. Safe DOM construction (HR-08).

// ============================================================================
// INIT
// ============================================================================

function initQuickActions() {
    var buttons = document.querySelectorAll('[data-insight-action]');
    buttons.forEach(function(btn) {
        btn.addEventListener('click', function() {
            var action = btn.getAttribute('data-insight-action');
            fetchInsight(action);
        });
    });
}

// ============================================================================
// FETCH + DISPATCH
// ============================================================================

function fetchInsight(actionName) {
    var resultsDiv = document.getElementById('insight-results');
    if (!resultsDiv) return;

    // Show loading
    resultsDiv.innerHTML = '';
    var spinner = document.createElement('div');
    spinner.className = 'text-center py-3';
    spinner.innerHTML = '<div class="spinner-border spinner-border-sm text-primary"></div> Loading...';
    resultsDiv.appendChild(spinner);

    // Highlight active button
    document.querySelectorAll('[data-insight-action]').forEach(function(b) {
        b.classList.remove('active');
    });
    var activeBtn = document.querySelector('[data-insight-action="' + actionName + '"]');
    if (activeBtn) activeBtn.classList.add('active');

    fetch('/api/v3/insights/' + actionName)
        .then(function(r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        })
        .then(function(data) {
            resultsDiv.innerHTML = '';
            renderInsightResult(actionName, data, resultsDiv);
        })
        .catch(function(e) {
            resultsDiv.innerHTML = '';
            var alert = document.createElement('div');
            alert.className = 'alert alert-danger small';
            alert.textContent = 'Failed to fetch insights: ' + e.message;
            resultsDiv.appendChild(alert);
        });
}

function renderInsightResult(action, data, container) {
    if (data.count === 0 && action !== 'health') {
        var empty = document.createElement('div');
        empty.className = 'text-muted small py-2';
        empty.textContent = 'No results for "' + action.replace(/_/g, ' ') + '".';
        container.appendChild(empty);
        return;
    }

    // Header with close button
    var header = document.createElement('div');
    header.className = 'd-flex justify-content-between align-items-center mb-2';
    var title = document.createElement('strong');
    title.className = 'small';
    title.textContent = formatActionName(action) + ' (' + data.count + ')';
    header.appendChild(title);
    var closeBtn = document.createElement('button');
    closeBtn.className = 'btn-close btn-close-sm';
    closeBtn.setAttribute('aria-label', 'Close');
    closeBtn.addEventListener('click', clearInsightResults);
    header.appendChild(closeBtn);
    container.appendChild(header);

    var renderers = {
        changed_this_week: renderChangedThisWeek,
        opinions: renderOpinions,
        contradictions: renderContradictions,
        health: renderHealth,
        cross_project: renderCrossProject,
    };
    var renderer = renderers[action];
    if (renderer) renderer(data, container);
}

// ============================================================================
// RENDERERS (safe DOM — createElement/textContent only, no innerHTML)
// ============================================================================

function renderChangedThisWeek(data, container) {
    var nodeIds = [];
    (data.items || []).forEach(function(item) {
        var card = document.createElement('div');
        card.className = 'border rounded p-2 mb-1';
        card.style.fontSize = '0.78rem';

        // Q5 (2026-10-06): show the memory kind (the same label the
        // Memories table and recall use), not the raw legacy fact_type.
        // memory_kind_label is absent only on a legacy pre-4.1.19 store
        // that has no kind columns at all, in which case the server never
        // sends it and this falls back to the old fact_type display.
        var badge = document.createElement('span');
        badge.className = 'badge bg-info me-1';
        badge.textContent = item.memory_kind_label || item.fact_type || 'fact';
        card.appendChild(badge);

        var content = document.createElement('span');
        content.textContent = item.content;
        card.appendChild(content);

        var ts = document.createElement('div');
        ts.className = 'text-muted mt-1';
        ts.style.fontSize = '0.7rem';
        ts.textContent = formatTimestamp(item.created_at);
        card.appendChild(ts);

        container.appendChild(card);
        if (item.fact_id) nodeIds.push(item.fact_id);
    });
    highlightNodesInGraph(nodeIds);
}

function renderOpinions(data, container) {
    var nodeIds = [];
    (data.items || []).forEach(function(item) {
        var card = document.createElement('div');
        card.className = 'border-start border-3 border-warning p-2 mb-1';
        card.style.fontSize = '0.78rem';

        var conf = document.createElement('span');
        conf.className = 'badge bg-warning text-dark me-1';
        conf.textContent = (item.confidence * 100).toFixed(0) + '%';
        card.appendChild(conf);

        var content = document.createElement('span');
        content.textContent = item.content;
        card.appendChild(content);

        var ts = document.createElement('div');
        ts.className = 'text-muted mt-1';
        ts.style.fontSize = '0.7rem';
        ts.textContent = formatTimestamp(item.created_at);
        card.appendChild(ts);

        container.appendChild(card);
        if (item.fact_id) nodeIds.push(item.fact_id);
    });
    highlightNodesInGraph(nodeIds);
}

function renderContradictions(data, container) {
    var nodeIds = [];
    (data.items || []).forEach(function(item) {
        var card = document.createElement('div');
        card.className = 'alert alert-warning d-flex align-items-start mb-2 py-2';

        var badge = document.createElement('span');
        badge.className = 'badge bg-danger me-2';
        badge.textContent = (item.severity * 100).toFixed(0) + '%';
        card.appendChild(badge);

        var body = document.createElement('div');
        body.style.fontSize = '0.78rem';

        var src = document.createElement('div');
        src.textContent = item.source_content;
        body.appendChild(src);

        var vs = document.createElement('strong');
        vs.className = 'text-danger';
        vs.textContent = ' vs ';
        body.appendChild(vs);

        var tgt = document.createElement('div');
        tgt.textContent = item.target_content;
        body.appendChild(tgt);

        card.appendChild(body);
        container.appendChild(card);

        if (item.source_id) nodeIds.push(item.source_id);
        if (item.target_id) nodeIds.push(item.target_id);
    });
    highlightNodesInGraph(nodeIds);
}

function renderHealth(data, container) {
    var health = (data.items && data.items[0]) || {};

    // Trust section
    var trust = health.trust || {};
    var trustCard = document.createElement('div');
    trustCard.className = 'border rounded p-2 mb-2';
    var trustTitle = document.createElement('strong');
    trustTitle.className = 'small';
    trustTitle.textContent = 'Trust Distribution';
    trustCard.appendChild(trustTitle);
    ['high', 'medium', 'low'].forEach(function(level) {
        var row = document.createElement('div');
        row.className = 'd-flex justify-content-between small';
        var label = document.createElement('span');
        label.textContent = level.charAt(0).toUpperCase() + level.slice(1);
        row.appendChild(label);
        var val = document.createElement('span');
        val.className = 'text-muted';
        val.textContent = trust[level] || 0;
        row.appendChild(val);
        trustCard.appendChild(row);
    });
    container.appendChild(trustCard);

    // Retention zones
    var zones = health.retention_zones;
    if (zones) {
        var retCard = document.createElement('div');
        retCard.className = 'border rounded p-2 mb-2';
        var retTitle = document.createElement('strong');
        retTitle.className = 'small';
        retTitle.textContent = 'Retention Zones';
        retCard.appendChild(retTitle);
        Object.keys(zones).forEach(function(zone) {
            var row = document.createElement('div');
            row.className = 'd-flex justify-content-between small';
            var label = document.createElement('span');
            label.textContent = zone;
            row.appendChild(label);
            var val = document.createElement('span');
            val.className = 'text-muted';
            val.textContent = zones[zone].count + ' (' + (zones[zone].avg_retention * 100).toFixed(0) + '%)';
            row.appendChild(val);
            retCard.appendChild(row);
        });
        container.appendChild(retCard);
    }

    // Totals
    var totals = health.totals || {};
    var totCard = document.createElement('div');
    totCard.className = 'border rounded p-2 mb-2';
    var totTitle = document.createElement('strong');
    totTitle.className = 'small';
    totTitle.textContent = 'Memory Totals';
    totCard.appendChild(totTitle);
    [['Facts', totals.facts], ['Entities', totals.entities], ['Edges', totals.edges], ['Communities', health.community_count || 0]].forEach(function(pair) {
        var row = document.createElement('div');
        row.className = 'd-flex justify-content-between small';
        var label = document.createElement('span');
        label.textContent = pair[0];
        row.appendChild(label);
        var val = document.createElement('span');
        val.className = 'text-muted';
        val.textContent = pair[1] || 0;
        row.appendChild(val);
        totCard.appendChild(row);
    });
    container.appendChild(totCard);
}

function renderCrossProject(data, container) {
    (data.items || []).forEach(function(item) {
        var card = document.createElement('div');
        card.className = 'border rounded p-2 mb-1';
        card.style.fontSize = '0.78rem';

        var name = document.createElement('strong');
        name.textContent = item.canonical_name;
        card.appendChild(name);

        var badge = document.createElement('span');
        badge.className = 'badge bg-primary ms-1';
        badge.textContent = item.session_count + ' sessions';
        card.appendChild(badge);

        var detail = document.createElement('div');
        detail.className = 'text-muted mt-1';
        detail.style.fontSize = '0.7rem';
        detail.textContent = (item.entity_type || 'entity') + ' — ' + item.fact_count + ' facts';
        card.appendChild(detail);

        container.appendChild(card);
    });
}

// ============================================================================
// EVENT BUS INTEGRATION (fire-and-forget — Phase 3 adds listener)
// ============================================================================

function highlightNodesInGraph(nodeIds) {
    if (!nodeIds || nodeIds.length === 0) return;
    window.dispatchEvent(new CustomEvent('slm:graph:highlight', {
        detail: { nodeIds: nodeIds },
    }));
}

function clearInsightResults() {
    var resultsDiv = document.getElementById('insight-results');
    if (resultsDiv) resultsDiv.innerHTML = '';
    // Clear button active state
    document.querySelectorAll('[data-insight-action]').forEach(function(b) {
        b.classList.remove('active');
    });
    // Clear graph highlights
    window.dispatchEvent(new CustomEvent('slm:graph:clearHighlight', { detail: {} }));
}

// ============================================================================
// HELPERS
// ============================================================================

function formatActionName(action) {
    return action.replace(/_/g, ' ').replace(/\b\w/g, function(c) { return c.toUpperCase(); });
}

function formatTimestamp(ts) {
    if (!ts) return '';
    try {
        var d = new Date(ts);
        return d.toLocaleDateString() + ' ' + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (_) {
        return ts;
    }
}

// ============================================================================
// AUTO-INIT
// ============================================================================

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initQuickActions);
} else {
    initQuickActions();
}
