/**
 * Enhanced Interactions for CIR App
 * Adds modern visual effects and micro-interactions
 */

(function() {
    'use strict';

    // Wait for DOM to be ready
    document.addEventListener('DOMContentLoaded', function() {
        initializeEnhancedInteractions();
    });

    function initializeEnhancedInteractions() {
        // Add fade-in animation to main components
        addFadeInAnimations();
        
        // Enhanced card hover effects
        enhanceCardInteractions();
        
        // Add particle effects to buttons
        enhanceButtonInteractions();
        
        // Add smooth scrolling
        enhanceSmoothScrolling();
        
        // Add loading state enhancements
        enhanceLoadingStates();
        
        // Add focus improvements
        enhanceFocusStates();
        
        // Initialize theme adaptations
        initializeThemeAdaptations();
    }

    function addFadeInAnimations() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in');
                    observer.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });

        // Observe main content areas
        const elementsToAnimate = document.querySelectorAll(
            '.border-widget, .stretchy-widget, .result-card, .gallery-card, .prompt-enhancement-card'
        );
        
        elementsToAnimate.forEach(el => {
            observer.observe(el);
        });
    }

    function enhanceCardInteractions() {
        // Add magnetic effect to cards (excluding prompt enhancement cards for simpler effect)
        // Prompt enhancement cards get a simplified hover effect to prevent overflow issues in fullscreen mode
        document.addEventListener('mousemove', function(e) {
            const cards = document.querySelectorAll('.result-card, .gallery-card');
            
            cards.forEach(card => {
                const rect = card.getBoundingClientRect();
                const x = e.clientX - rect.left;
                const y = e.clientY - rect.top;
                
                if (x >= 0 && x <= rect.width && y >= 0 && y <= rect.height) {
                    const centerX = rect.width / 2;
                    const centerY = rect.height / 2;
                    const deltaX = (x - centerX) / centerX;
                    const deltaY = (y - centerY) / centerY;
                    
                    const tiltX = deltaY * 5; // Max 5 degrees
                    const tiltY = deltaX * -5;
                    
                    card.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateZ(10px)`;
                }
            });
        });

        // Reset transform when mouse leaves
        document.addEventListener('mouseleave', function() {
            const cards = document.querySelectorAll('.result-card, .gallery-card');
            cards.forEach(card => {
                card.style.transform = '';
            });
        });

        // Add simpler, more stable hover effect for prompt enhancement cards
        document.addEventListener('mouseover', function(e) {
            const promptCard = e.target.closest('.prompt-enhancement-card');
            if (promptCard) {
                // Check if we're in fullscreen mode - multiple detection methods
                const parentCard = promptCard.closest('#prompt-enh-card');
                const isFullscreen = parentCard && (
                    parentCard.style.position === 'fixed' || 
                    parentCard.style.width === '100vw' ||
                    parentCard.style.height === '100vh'
                );
                
                if (isFullscreen) {
                    // Very simple effect for fullscreen mode - just shadow
                    promptCard.style.boxShadow = '0 4px 15px rgba(0, 0, 0, 0.1)';
                    promptCard.style.transition = 'box-shadow 0.2s ease';
                } else {
                    // Clean hover effect for normal mode - no scaling to prevent blurriness
                    promptCard.style.transform = 'translateY(-3px)';
                    promptCard.style.boxShadow = '0 8px 25px rgba(0, 0, 0, 0.15)';
                    promptCard.style.transition = 'transform 0.2s ease, box-shadow 0.2s ease';
                }
            }
        });

        document.addEventListener('mouseout', function(e) {
            const promptCard = e.target.closest('.prompt-enhancement-card');
            if (promptCard) {
                // Reset to original state
                promptCard.style.transform = '';
                promptCard.style.boxShadow = '';
                promptCard.style.transition = '';
            }
        });
    }

    function enhanceButtonInteractions() {
        // Add ripple effect to buttons
        document.addEventListener('click', function(e) {
            if (e.target.matches('.btn') || e.target.closest('.btn')) {
                const button = e.target.matches('.btn') ? e.target : e.target.closest('.btn');
                createRippleEffect(button, e);
            }
        });

        function createRippleEffect(button, event) {
            const ripple = document.createElement('span');
            const rect = button.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            const x = event.clientX - rect.left - size / 2;
            const y = event.clientY - rect.top - size / 2;
            
            ripple.style.cssText = `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple 0.6s linear;
                left: ${x}px;
                top: ${y}px;
                width: ${size}px;
                height: ${size}px;
                pointer-events: none;
            `;
            
            // Add ripple animation keyframes if not already added
            if (!document.querySelector('#ripple-styles')) {
                const style = document.createElement('style');
                style.id = 'ripple-styles';
                style.textContent = `
                    @keyframes ripple {
                        to {
                            transform: scale(2);
                            opacity: 0;
                        }
                    }
                `;
                document.head.appendChild(style);
            }
            
            // Ensure button has relative positioning
            if (getComputedStyle(button).position === 'static') {
                button.style.position = 'relative';
            }
            button.style.overflow = 'hidden';
            
            button.appendChild(ripple);
            
            setTimeout(() => {
                ripple.remove();
            }, 600);
        }
    }

    function enhanceSmoothScrolling() {
        // Add smooth scrolling to all scroll containers
        const scrollContainers = document.querySelectorAll('.gallery, .wordcloud-container, [style*="overflow"]');
        
        scrollContainers.forEach(container => {
            container.style.scrollBehavior = 'smooth';
        });
    }

    function enhanceLoadingStates() {
        // Enhanced loading animations
        const observer = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === 1) { // Element node
                        // Check for spinners
                        const spinners = node.querySelectorAll('.spinner-border, .spinner-grow');
                        spinners.forEach(spinner => {
                            spinner.style.filter = 'drop-shadow(0 0 10px rgba(102, 126, 234, 0.5))';
                        });
                        
                        // Add pulse animation to loading containers
                        if (node.textContent && node.textContent.includes('loading') || 
                            node.textContent.includes('Loading') ||
                            node.textContent.includes('generating') ||
                            node.textContent.includes('Generating')) {
                            node.style.animation = 'pulse 2s infinite';
                        }
                    }
                });
            });
        });

        observer.observe(document.body, { childList: true, subtree: true });

        // Add pulse animation if not exists
        if (!document.querySelector('#loading-styles')) {
            const style = document.createElement('style');
            style.id = 'loading-styles';
            style.textContent = `
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.7; }
                }
            `;
            document.head.appendChild(style);
        }
    }

    function enhanceFocusStates() {
        // Add enhanced focus indicators
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                document.body.classList.add('keyboard-navigation');
            }
        });

        document.addEventListener('mousedown', function() {
            document.body.classList.remove('keyboard-navigation');
        });

        // Add focus styles for keyboard navigation
        if (!document.querySelector('#focus-styles')) {
            const style = document.createElement('style');
            style.id = 'focus-styles';
            style.textContent = `
                .keyboard-navigation *:focus {
                    outline: 3px solid rgba(102, 126, 234, 0.6) !important;
                    outline-offset: 2px !important;
                    border-radius: 8px !important;
                }
            `;
            document.head.appendChild(style);
        }
    }

    function initializeThemeAdaptations() {
        // Check for dark mode preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)');
        
        function updateTheme(e) {
            if (e.matches) {
                document.body.classList.add('dark-mode');
            } else {
                document.body.classList.remove('dark-mode');
            }
        }

        prefersDark.addListener(updateTheme);
        updateTheme(prefersDark);

        // Add reduced motion support
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        
        function updateMotion(e) {
            if (e.matches) {
                document.body.classList.add('reduced-motion');
            } else {
                document.body.classList.remove('reduced-motion');
            }
        }

        prefersReducedMotion.addListener(updateMotion);
        updateMotion(prefersReducedMotion);
    }

    // Enhanced hover effects for specific elements
    function addGlowEffect(element) {
        element.addEventListener('mouseenter', function() {
            this.style.filter = 'drop-shadow(0 0 20px rgba(102, 126, 234, 0.4))';
        });

        element.addEventListener('mouseleave', function() {
            this.style.filter = '';
        });
    }

    // Parallax effect for background elements
    function initializeParallax() {
        const parallaxElements = document.querySelectorAll('[data-parallax]');
        
        if (parallaxElements.length > 0) {
            window.addEventListener('scroll', function() {
                const scrolled = window.pageYOffset;
                
                parallaxElements.forEach(element => {
                    const rate = scrolled * -0.5;
                    element.style.transform = `translateY(${rate}px)`;
                });
            });
        }
    }

    // Initialize additional enhancements after a short delay
    setTimeout(() => {
        initializeParallax();
        
        // Add glow effect to important elements
        const importantElements = document.querySelectorAll('.btn-primary, .selected-gallery-card, .visual-selected');
        importantElements.forEach(addGlowEffect);
        
        // Add stagger animation to grid items
        const gridItems = document.querySelectorAll('.result-card-wrapper, .gallery-card');
        gridItems.forEach((item, index) => {
            item.style.animationDelay = `${index * 0.1}s`;
            item.classList.add('fade-in');
        });
    }, 500);

    // Performance optimizations
    let ticking = false;
    
    function optimizedScrollHandler() {
        if (!ticking) {
            requestAnimationFrame(() => {
                // Perform scroll-based animations here
                ticking = false;
            });
            ticking = true;
        }
    }

    window.addEventListener('scroll', optimizedScrollHandler);

    // Add error boundary for graceful degradation
    window.addEventListener('error', function(e) {
        console.warn('Enhanced interactions error:', e.error);
        // Gracefully continue without enhanced features
    });

})();

/**
 * Utility functions for enhanced interactions
 */
window.CIREnhancements = {
    // Function to manually trigger fade-in animation
    triggerFadeIn: function(selector) {
        const elements = document.querySelectorAll(selector);
        elements.forEach(el => el.classList.add('fade-in'));
    },
    
    // Function to add custom glow effect
    addGlow: function(element, color = 'rgba(102, 126, 234, 0.4)') {
        element.style.filter = `drop-shadow(0 0 20px ${color})`;
    },
    
    // Function to remove glow effect
    removeGlow: function(element) {
        element.style.filter = '';
    },
    
    // Function to create custom notification
    showNotification: function(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `alert alert-${type} position-fixed`;
        notification.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 9999;
            min-width: 300px;
            animation: slideInRight 0.5s ease-out;
        `;
        notification.innerHTML = `
            <i class="fas fa-info-circle me-2"></i>
            ${message}
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.5s ease-in';
            setTimeout(() => notification.remove(), 500);
        }, 3000);
    }
}; 