(function(){
    // Utility to position tooltip relative to cursor
    function positionTooltip(evt, tooltip){
        const tooltipRect = tooltip.getBoundingClientRect();
        let top = evt.pageY - tooltipRect.height - 14; // 14px above cursor
        let left = evt.pageX - (tooltipRect.width / 2);

        const scrollY = window.scrollY || document.documentElement.scrollTop;
        const scrollX = window.scrollX || document.documentElement.scrollLeft;
        const viewportWidth = document.documentElement.clientWidth;
        const viewportHeight = document.documentElement.clientHeight;

        // Constrain horizontally
        if(left < scrollX + 6) left = scrollX + 6;
        if(left + tooltipRect.width > scrollX + viewportWidth - 6){
            left = scrollX + viewportWidth - tooltipRect.width - 6;
        }

        // If not enough space above, place below cursor
        if(top < scrollY + 6){
            top = evt.pageY + 14;
            if(top + tooltipRect.height > scrollY + viewportHeight - 6){
                // Still overflowing, clamp
                top = scrollY + viewportHeight - tooltipRect.height - 6;
            }
        }

        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
    }

    // Event delegation: listen for mouseover on document
    document.addEventListener('mouseover', function(e){
        const container = e.target.closest('.tooltip-container');
        if(!container) return;
        const tooltip = container.querySelector('.custom-tooltip');
        if(!tooltip) return;
        tooltip.style.visibility = 'visible';
        tooltip.style.opacity = '1';
        positionTooltip(e, tooltip);

        // Follow cursor movement
        const moveListener = (ev) => positionTooltip(ev, tooltip);
        container.addEventListener('mousemove', moveListener);

        const leaveListener = () => {
            tooltip.style.visibility = 'hidden';
            tooltip.style.opacity = '0';
            container.removeEventListener('mousemove', moveListener);
            container.removeEventListener('mouseleave', leaveListener);
        };
        container.addEventListener('mouseleave', leaveListener);
    });
})(); 