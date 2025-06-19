(function () {
  // Helper: scroll a specific container to centre given row
  function centreRowInContainer(row, container) {
    if (!row || !container) return;
    // Bring into view using browser logic, then ensure it's roughly centred
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    // Additional centring tweak
    const containerRect = container.getBoundingClientRect();
    const rowRect = row.getBoundingClientRect();
    const offset = rowRect.top - containerRect.top;
    const desired = container.scrollTop + offset - (container.clientHeight / 2) + (rowRect.height / 2);
    container.scrollTo({ top: desired, behavior: 'smooth' });
  }

  function scrollAll() {
    console.debug('[RankΔ scroll] scrollAll triggered');
    const containers = document.querySelectorAll('#rank-delta-content');
    containers.forEach((container) => {
      const selectedRow = container.querySelector('tr.table-info');
      if (selectedRow) {
        console.debug('[RankΔ scroll] Found selected row in container', container);
        centreRowInContainer(selectedRow, container);
      } else {
        // If no selected row, optionally scroll to top
        // container.scrollTo({ top: 0, behavior: 'smooth' });
      }
    });
  }

  function attachObservers() {
    const containers = document.querySelectorAll('#rank-delta-content');
    if (!containers.length) {
      console.debug('[RankΔ scroll] No containers yet, retrying…');
      setTimeout(attachObservers, 500);
      return;
    }

    containers.forEach((container) => {
      const obs = new MutationObserver(scrollAll);
      obs.observe(container, { childList: true, subtree: true, attributes: true, attributeFilter: ['class'] });
    });
    // Initial scroll attempt
    setTimeout(scrollAll, 200);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', attachObservers);
  } else {
    attachObservers();
  }
})(); 