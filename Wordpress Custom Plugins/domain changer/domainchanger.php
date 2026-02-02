<?php
/*
Plugin Name: Domain Viewer
Description: Displays a user-defined domain on the front end.
Version: 1.1
Author: Your Name
*/

// Exit if accessed directly
if ( ! defined( 'ABSPATH' ) ) {
    exit;
}

add_action( 'admin_menu', 'domain_viewer_add_menu_page' );

function domain_viewer_add_menu_page() {
    add_options_page(
        'Domain Viewer Settings',
        'Domain Viewer',
        'manage_options',
        'domain-viewer',
        'domain_viewer_options_page_html'
    );
}

function domain_viewer_options_page_html() {
    if ( ! current_user_can( 'manage_options' ) ) {
        return;
    }

    if ( isset( $_POST['domain_viewer_domain_field'] ) ) {
        $domain = sanitize_text_field( $_POST['domain_viewer_domain_field'] );
        update_option( 'domain_viewer_domain_url', $domain );
        echo '<div class="notice notice-success is-dismissible"><p>Domain saved successfully!</p></div>';
    }

    $saved_domain = get_option( 'domain_viewer_domain_url', '' );
    ?>
    <div class="wrap">
        <h1>Domain Viewer Settings</h1>
        <form action="" method="post">
            <table class="form-table">
                <tbody>
                    <tr>
                        <th scope="row"><label for="domain_viewer_domain_field">Enter Domain URL</label></th>
                        <td>
                            <input name="domain_viewer_domain_field" type="url" id="domain_viewer_domain_field" class="regular-text" value="<?php echo esc_url( $saved_domain ); ?>">
                            <p class="description">e.g., https://example.com</p>
                        </td>
                    </tr>
                </tbody>
            </table>
            <p class="submit">
                <input type="submit" name="submit" id="submit" class="button button-primary" value="Save Domain">
            </p>
        </form>
    </div>
    <?php
}

add_action( 'wp_enqueue_scripts', 'domain_viewer_enqueue_scripts' );

function domain_viewer_enqueue_scripts() {
    $domain_url = get_option( 'domain_viewer_domain_url', '' );

    if ( ! empty( $domain_url ) ) {
        // First, register a script handle. It doesn't need a source file.
        wp_register_script(
            'domain-viewer-script-handle',
            '', // No source file
            array(), // No dependencies
            '1.0', // Version
            true // Load in footer
        );

        // Now, add the inline script to the registered handle.
        wp_add_inline_script(
            'domain-viewer-script-handle',
            'var myCustomDomain = "' . esc_js( $domain_url ) . '";'
        );

        // Finally, enqueue the script to make sure it's loaded.
        wp_enqueue_script( 'domain-viewer-script-handle' );
    }
}