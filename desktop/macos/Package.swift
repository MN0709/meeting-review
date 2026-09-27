// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "MeetingReviewDesktop",
    platforms: [.macOS(.v13)],
    products: [
        .executable(name: "MeetingReviewDesktop", targets: ["MeetingReviewDesktop"]),
    ],
    targets: [
        .executableTarget(
            name: "MeetingReviewDesktop",
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("Security"),
                .linkedFramework("WebKit"),
            ]
        ),
    ]
)
