import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import QtQuick.Controls.Material 2.15

ApplicationWindow {
    id: window
    visible: true
    width: 900
    height: 700
    title: "🎵 Song Genre Classifier"
    
    Material.theme: Material.Light
    Material.accent: Material.Blue
    
    // Background gradient
    background: Rectangle {
        gradient: Gradient {
            GradientStop { position: 0.0; color: "#f5f7fa" }
            GradientStop { position: 1.0; color: "#c3cfe2" }
        }
    }
    
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 20
        
        // Title
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 80
            color: "#2c3e50"
            radius: 15
            
            RowLayout {
                anchors.centerIn: parent
                spacing: 15
                
                Text {
                    text: "🎵"
                    font.pixelSize: 48
                }
                
                Text {
                    text: "Song Genre Classifier"
                    font.pixelSize: 32
                    font.bold: true
                    color: "#ecf0f1"
                }
            }
            
            // Animated gradient effect
            layer.enabled: true
            layer.effect: ShaderEffect {
                property real time: 0
                NumberAnimation on time {
                    from: 0
                    to: 1
                    duration: 3000
                    loops: Animation.Infinite
                }
            }
        }
        
        // File Selection Card
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 120
            color: "white"
            radius: 12
            border.color: "#3498db"
            border.width: 2
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 20
                spacing: 10
                
                Text {
                    text: "Select Audio File"
                    font.pixelSize: 16
                    font.bold: true
                    color: "#2c3e50"
                }
                
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 15
                    
                    Rectangle {
                        Layout.fillWidth: true
                        Layout.preferredHeight: 50
                        color: "#ecf0f1"
                        radius: 8
                        
                        Text {
                            anchors.centerIn: parent
                            text: backend.fileName
                            font.pixelSize: 14
                            color: "#34495e"
                            elide: Text.ElideMiddle
                            width: parent.width - 20
                        }
                    }
                    
                    Button {
                        Layout.preferredWidth: 120
                        Layout.preferredHeight: 50
                        text: "Browse"
                        font.pixelSize: 14
                        font.bold: true
                        
                        Material.background: Material.Blue
                        Material.foreground: "white"
                        
                        onClicked: backend.browseFile()
                        
                        // Hover effect
                        hoverEnabled: true
                        ToolTip.visible: hovered
                        ToolTip.text: "Select an audio file (MP3, WAV, FLAC, OGG, M4A)"
                    }
                }
            }
        }
        
        // Classify Button
        Button {
            Layout.fillWidth: true
            Layout.preferredHeight: 70
            text: "🎸 Classify Genre"
            font.pixelSize: 20
            font.bold: true
            enabled: backend.classifyEnabled
            
            Material.background: Material.Green
            Material.foreground: "white"
            
            onClicked: backend.classifyFile()
            
            // Scale animation on press
            scale: pressed ? 0.95 : 1.0
            Behavior on scale {
                NumberAnimation { duration: 100 }
            }
        }
        
        // Progress Bar
        ProgressBar {
            Layout.fillWidth: true
            Layout.preferredHeight: 8
            visible: backend.progressVisible
            indeterminate: true
            
            Material.accent: Material.Blue
        }
        
        // Status Label
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            color: "#3498db"
            radius: 8
            
            Text {
                anchors.centerIn: parent
                text: backend.status
                font.pixelSize: 14
                font.italic: true
                color: "white"
            }
        }
        
        // Results Card
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "white"
            radius: 12
            border.color: "#27ae60"
            border.width: 2
            
            ColumnLayout {
                anchors.fill: parent
                anchors.margins: 15
                spacing: 10
                
                Text {
                    text: "📊 Classification Results"
                    font.pixelSize: 18
                    font.bold: true
                    color: "#2c3e50"
                }
                
                Rectangle {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    color: "#2c3e50"
                    radius: 8
                    
                    ScrollView {
                        anchors.fill: parent
                        anchors.margins: 10
                        clip: true
                        
                        TextArea {
                            text: backend.resultText
                            font.family: "Courier New"
                            font.pixelSize: 13
                            color: "#ecf0f1"
                            readOnly: true
                            selectByMouse: true
                            wrapMode: TextEdit.NoWrap
                            background: Rectangle {
                                color: "transparent"
                            }
                        }
                    }
                }
            }
        }
        
        // Footer
        Rectangle {
            Layout.fillWidth: true
            Layout.preferredHeight: 50
            color: "#34495e"
            radius: 8
            
            ColumnLayout {
                anchors.centerIn: parent
                spacing: 2
                
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "Supports: MP3, WAV, FLAC, OGG, M4A"
                    font.pixelSize: 11
                    color: "#ecf0f1"
                }
                
                Text {
                    Layout.alignment: Qt.AlignHCenter
                    text: "10 Genres: Rock, Pop, Jazz, Classical, Hip-Hop, Electronic, Country, Metal, Blues, Reggae"
                    font.pixelSize: 11
                    color: "#bdc3c7"
                }
            }
        }
    }
    
    // Window animations
    NumberAnimation on opacity {
        from: 0
        to: 1
        duration: 500
        running: true
    }
}
